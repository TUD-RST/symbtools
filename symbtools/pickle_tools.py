import pickle
import sympy as sp
from collections import defaultdict
from .aux import Container, global_data
from .time_deriv import get_all_deriv_parents


# handling of _attribute_store makes custom pickle interface necessary
def pickle_full_dump(obj, path):
    """write sympy object (expr, matrix, ...) or Container object to file
    via pickle serialization and additionally also dump the corresponding
    entries of _attribute_store (such as difforder).
    """

    if isinstance(obj, Container):
        pdata = obj

        # prevent accidental name clashes
        assert not hasattr(pdata, 'relevant_symbols')
        assert not hasattr(pdata, 'attribute_store')
        assert not hasattr(pdata, 'atoms')
        assert not hasattr(pdata, 'data')

        pdata.container_flag = True
        additional_data = pdata

    elif isinstance(obj, sp.MatrixBase):

        pdata = Container()

        pdata.container_flag = False
        # safe obj so that it will be pickled
        pdata.obj = obj

        if hasattr(obj, 'data'):
            assert isinstance(obj.data, Container)
            additional_data = obj.data
        else:
            additional_data = None
    else:
        raise TypeError('Unexpected data type: %s' % type(obj))

    pdata.relevant_symbols = list()

    # helper function:
    def get_symbols(my_obj):
        if hasattr(my_obj, 'atoms'):
            pdata.relevant_symbols += list(my_obj.atoms(sp.Symbol))

    # apply that function to obj itself
    get_symbols(obj)

    # now apply it to all items in additional_data

    if additional_data:
        for new_obj in list(additional_data.__dict__.values()):
            get_symbols(new_obj)

    # make each symbol occur only once
    pdata.relevant_symbols = set(pdata.relevant_symbols)

    # find out which symbol names occur more than once in the set
    # this indicates that there are symbols with the same name but different
    # assumptions (like commutativity)
    # due to strange interaction of sympy and pickle this leads to unexpected results
    # after unpickling
    symbol_names = [s.name for s in pdata.relevant_symbols]
    unique_names = set(symbol_names)

    multiple_name_count = []
    for u in unique_names:
        count = symbol_names.count(u)
        if count > 1:
            multiple_name_count.append((u, count))

    if len(multiple_name_count) > 0:
        msg = "The following symbol names occur more than once but have different assumptions "\
              "(such as `commutative=False`): "
        msg += str(multiple_name_count)

        raise ValueError(msg)

    # now look in global_data.attribute_store (see above) if there are
    # some attributes stored concerning the relevant_symbols
    # global_data.attribute_store looks like {(xdot, 'difforder'): 1, ...}
    relevant_attributes, function_keys = find_relevant_attributes(pdata.relevant_symbols)

    # to this end collect every attribute of pdata which is as (generalized) sympy expression

    substituted_attributes, function_data = replace_functions_in_pdata(pdata, relevant_attributes, function_keys)

    pdata.attribute_store = substituted_attributes
    pdata.function_data = function_data

    # explicitly save additional data (because the custom attribute seems not to be preserved by
    # pickling)
    pdata.additional_data = additional_data

    with open(path, 'wb') as pfile:
        pickle.dump(pdata, pfile)


class PseudoAppliedFunc(object):
    def __init__(self, appl_func):
        """
        store the relevant Information of an instance of AppliedUndef (which cannot be pickled):
        name, args, assumptions

        :param appl_func:
        """

        assert isinstance(appl_func, sp.function.AppliedUndef)
        self.name = appl_func.name
        self.args = appl_func.args
        # noinspection PyProtectedMember
        self.assumptions = appl_func._assumptions

        if not appl_func.atoms(sp.function.AppliedUndef) == {appl_func}:
            msg = "nested calls of applied Functions are not yet supported."
            raise NotImplementedError(msg)

    def make_func(self):

        return sp.Function(self.name, **self.assumptions)(*self.args)


def convert_functions_to_symbols(appl_func_list):
    """
    Because unnamed functions cannot be pickled or dilled, we convert them to symbols before pickling,
    and convert them back after unpickling.

    :param appl_func_list:  list of applied functions like ([x1(t), x3(t), ...])

    :return:    rplmts, function_data

    rplmts: list of 2-tuples
    function_data: dict like {_FUNC0: <PseudoAppliedFunc instandce0>, ...}
    """

    rplmts = []
    function_data = {}
    for i, f in enumerate(appl_func_list):
        # noinspection PyProtectedMember
        func_symb = sp.Dummy("FUNC{}".format(i), **f._assumptions)
        rplmts.append((f, func_symb))
        function_data[func_symb] = PseudoAppliedFunc(f)

    return rplmts, function_data


def find_relevant_attributes(symbol_list):

    working_list = list(symbol_list)
    known_symbols = set(symbol_list)

    # dict of all relevant attributes
    relevant_attributes = {}

    # find out in which keys there are AppliedUndef-Instances
    function_keys = defaultdict(set)

    # global_data.attribute_store looks like {(xddot, 'difforder'): 2, ...}
    ga_items = global_data.attribute_store.items()

    while len(working_list) > 0:
        s = working_list.pop(0)
        local_relevant_items = [item for item in ga_items if item[0][0] == s]
        relevant_attributes.update(local_relevant_items)

        new_symbol_candidates = set()
        for key, value in local_relevant_items:
            if hasattr(value, "atoms"):
                symbs = value.atoms(sp.Symbol)
                appl_funcs = value.atoms(sp.function.AppliedUndef)
                new_symbol_candidates = new_symbol_candidates.union(symbs)

                # store the key to easily access this attribute later
                for f in appl_funcs:
                    function_keys[f].add(key)

        for cand in new_symbol_candidates:
            if cand not in known_symbols:
                working_list.append(cand)
                known_symbols.add(cand)

    return relevant_attributes, function_keys


def replace_functions_in_pdata(pdata, attributes, function_keys):
    """

    :param pdata:           container of which some attributes are sympy expressions (will be altered)
    :param attributes:      relevant attribute-dict, in which the functions have to be replaced
    :param function_keys:   dict like {x1(t): {key1, key2}} (where key1 etc refer to the attributes-dict)

    :return:    replaced_expr, replaced_attributes, function_data
    """

    sp_obj_keys = []
    expr_funcs = set()  # all functions which occur in expr-attributes of pdata
    for key, value in pdata.__dict__.items():
        if isinstance(value, (sp.Expr, sp.MatrixBase)):
            sp_obj_keys.append(key)
            expr_funcs.update(value.atoms(sp.function.AppliedUndef))

    attr_funcs = set(function_keys.keys())  # functions which are hidden in attribute-store dict

    all_funcs = expr_funcs.union(attr_funcs)

    rplmts, function_data = convert_functions_to_symbols(all_funcs)

    for key in sp_obj_keys:
        pdata.__dict__[key] = pdata.__dict__[key].subs(rplmts)

    substiuted_attributes = dict(attributes)

    united_key_set = set()

    for applied_func, keyset in function_keys.items():
        united_key_set.update(keyset)

    # apply replacements
    for key in united_key_set:
        substiuted_attributes[key] = substiuted_attributes[key].subs(rplmts)

    return substiuted_attributes, function_data


def get_rplmts_from_function_data(function_data):
    """

    :param function_data:   dict like {_FUNC0: <PseudoAppliedFunc instandce0>, ...}

    :return:    list of 2-tuples like [(_FUNC0, x1(t)), ...] (where x1(t) comes from PseudoAppliedFunc.make_func())
    """

    rplmts = []
    for symbol, pseudo_func in function_data.items():
        rplmts.append((symbol, pseudo_func.make_func()))

    return rplmts


def pickle_full_load(path):
    """load sympy object (expr, matrix, ...) or Container object from file
    via pickle serialization and additionally also load the corresponding
    entries of _attribute_store (such as difforder).
    """

    with open(path, 'rb') as pfile:
        pdata = pickle.load(pfile)

    new_items = list(pdata.attribute_store.items())

    # handle functions (which have been converted to symbols to allow pickling)
    if not hasattr(pdata, 'function_data'):
        pdata.function_data = {}

    rplmts = get_rplmts_from_function_data(pdata.function_data)
    new_attributes = {}

    for key, value in new_items:

        # apply replacements
        if hasattr(value, "subs"):
            value = value.subs(rplmts)
        new_attributes[key] = value

        # prevent silent overwriting of attributes
        old_value = global_data.attribute_store.get(key)
        if old_value is None or value == old_value:
            continue
        msg = "Name conflict while loading attributes from serialization.\n"
        msg += "Attribute %s: \n old value: %s \n new value: %s" % (key, old_value, value)
        raise ValueError(msg)

    global_data.attribute_store.update(new_attributes)

    # allow to load older containers without that flag
    if not hasattr(pdata, 'container_flag'):
        return pdata

    try:
        pdata.obj = pdata.obj.subs(rplmts)
    except AttributeError:
        pass

    try:
        obj = pdata.obj
    except AttributeError:
        obj = None

    # manually set the (optional data-attribute for Matrices)
    if isinstance(obj, sp.MatrixBase) and getattr(pdata, "additional_data", None) is not None:
        obj.data = pdata.additional_data

    if pdata.container_flag:
        # return the whole container
        return pdata
    else:
        # return just that attribute
        return pdata.obj
