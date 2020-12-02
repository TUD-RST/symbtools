import numpy as np
import sympy as sp
import symbtools as st
import warnings
import colorsys

# The following packages are only necessary for visualization. The respective requirements are listed in
# visualization_requirements.txt

# noinspection PyPackageRequirements
import matplotlib.pyplot as plt
# noinspection PyPackageRequirements
import matplotlib.animation as animation
# noinspection PyPackageRequirements
import matplotlib.colors as colors
try:
    # TODO: the whole way we handle different usage contexts is very not great right now.
    #  For example plots get automatically closed, making it very hard to see them outside of Jupyter
    # noinspection PyPackageRequirements
    from IPython.display import HTML, display as ip_display
    # noinspection PyPackageRequirements
    from ipywidgets import FloatSlider, Checkbox, interact
    in_ipython_context = True
except ImportError:
    in_ipython_context = False
    HTML, ip_display, FloatSlider, interact = (ImportError("__object_not_imported__"),)*4

from scipy.optimize import fmin
from .core import expr_to_func


def merge_options(custom_options, **default_options):
    """
    Utility function to merge some default options with a dictionary of custom_options.
    Example: custom_options = dict(a=5, b=3)
             merge_options(custom_options, a=1, c=4)
             --> results in {a: 5, b: 3, c: 4}
    """
    merged_options = default_options
    merged_options.update(custom_options)
    return merged_options


class VisualiserElement:
    """
    One visual element in a kinematic visualisation, for example a link or a polygon.
    """
    def __init__(self, points_fun, init_fun, update_fun, kwargs):
        """
        :param points_fun: callable, that takes values for all visualiser variables as arguments and returns a 2x?
        matrix of 2D points that define the position/orientation of the element
        :param init_fun: callable with args (matplotlib axes, 2x? numpy array of points, dict of kwargs) that returns a
        list of matplotlib drawables. Will get called to create all drawables needed by this element.
        :param update_fun: callable with args (matplotlib axes, list of drawables, 2x? numpy array of points, dict of
        kwargs) that returns a list of matplotlib drawables. Will get called every time the plot needs to be updated.
        :param kwargs: dict of arbitrary keyword arguments that get passed to init_fun and update_fun
        """
        self.points_fun = points_fun
        self.init_fun = init_fun
        self.update_fun = update_fun
        self.kwargs = kwargs
        self.drawables = []
        """list of drawables created by this element, required to update their data when kinematic values changed"""


class Visualiser:
    def __init__(self, variables, **axes_kwargs):
        """
        Creates a new visualiser. A visualiser describes a set of graphical elements that somehow depend on some free
        variables. It can be used for plotting a kinematic configuration, creating an interactive version or rendering
        animations of a numeric simulation.
        :param variables: iterable of SymPy symbols, all free variables in the system to be visualised
        :param axes_kwargs: keyword arguments that should be passed to the axes object that is automatically created
        """
        self.variables = variables
        self.elements = []
        self.axes_kwargs = axes_kwargs

    def create_default_axes(self, fig=None, add_subplot_args=111):
        """
        Create a figure if none is given and add axes to it
        :param fig: the figure to add the axes to
        :param add_subplot_args: specification of the subplot layout that is passed to fig.add_subplot
        :return: (figure, axes)
        """
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(add_subplot_args, **merge_options(self.axes_kwargs, aspect='equal', xlim=(-2.0, 2.0), ylim=(-2.0, 2.0)))
        ax.grid()

        return fig, ax

    def plot(self, variables_values, axes=None):
        """
        Plot for some specific variable values.
        :param variables_values: iterable of values for all free variables
        :param axes: the matplotlib axes to plot on, one will be created if none is given
        """
        assert len(self.variables) == len(variables_values), \
            f"You need to pass as many variable values as this visualiser has variables. Required:" \
            f"{len(self.variables)}, Given: {len(variables_values)}"

        fig = None
        if axes is None:
            fig, axes = self.create_default_axes()
            # TODO: Probably should not do that, but it prevents an empty plot from popping up in IPython
            plt.close(fig)

        self.plot_init(variables_values, axes)
        self.plot_update(variables_values, axes)
        if fig is not None and in_ipython_context:
            ip_display(fig)

    def plot_init(self, variables_values, axes):
        """
        Initialize all graphical elements
        :param variables_values: iterable of values for all free variables
        :param axes: the matplotlib axes to plot on
        :return: list of created drawables
        """
        drawables = []
        for element in self.elements:
            element.drawables = element.init_fun(axes, element.points_fun(*variables_values), element.kwargs)
            drawables += element.drawables

        return drawables

    def plot_update(self, variables_values, axes):
        """
        Update all graphical elements with the current free variable values
        :param variables_values: iterable of values for all free variables
        :param axes: the matplotlib axes to plot on
        :return: list of updated drawables
        """
        drawables = []
        for element in self.elements:
            element.drawables = element.update_fun(axes, element.drawables, element.points_fun(*variables_values),
                                                   element.kwargs)
            drawables += element.drawables

        return drawables

    def plot_onion_skinned(self, variables_values, axes=None, change_alpha=False, max_lightness=0.9):
        """
        !!! EXPERIMENTAL !!!
        Plot multiple configurations in one picture with 'older' data shown lighter
        :param variables_values: 2D NumPy array, each row as one set of variable values
        :param axes: the matplotlib axes to plot on, one will be created if none is given
        """
        assert len(self.variables) == variables_values.shape[1], \
            f"You need to pass as many variable values as this visualiser has variables. Required:" \
            f"{len(self.variables)}, Given: {variables_values.shape[1]}"

        fig = None
        if axes is None:
            fig, axes = self.create_default_axes()
            # TODO: Probably should not do that, but it prevents an empty plot from popping up in IPython
            plt.close(fig)

        def lighten_color(orig_mpl_color, i_norm):
            orig_rgb = colors.to_rgb(orig_mpl_color)  # 3-tuple of floats [0, 1]
            # convert to hue, lightness, saturation color space
            orig_h, orig_l, orig_s = colorsys.rgb_to_hls(*orig_rgb)
            # interpolate, max_lightness for oldest frame, original lightness for newest
            #new_l = orig_l + (max_lightness - orig_l) * (1 - i_norm)  # linear interpolation
            new_l = (orig_l - max_lightness) * i_norm**2 + max_lightness  # quadratic, local max at higher brightness
            new_rgb = colorsys.hls_to_rgb(orig_h, new_l, orig_s)

            return new_rgb

        total_frames = variables_values.shape[0]
        
        for i in range(total_frames):
            frame_values = variables_values[i]
            i_norm = i / (total_frames - 1)  # 0.0 is iteration start, 1.0 is last iteration
            drawables = self.plot_init(frame_values, axes)
            for drawable in drawables:
                if change_alpha:
                    # interpolate alpha linearly from (1 - max_lightness) to 1
                    new_alpha = 1 - (1-i_norm) * max_lightness
                    drawable.set_alpha(new_alpha)
                else:
                    try:
                        # drawable is a line
                        new_color = lighten_color(drawable.get_color(), i_norm)
                        new_mec = lighten_color(drawable.get_mec(), i_norm)
                        new_mfc = lighten_color(drawable.get_mfc(), i_norm)
                        drawable.set_color(new_color)
                        drawable.set_mec(new_mec)
                        drawable.set_mfc(new_mfc)
                    except:
                        # drawable is a patch
                        new_fc = lighten_color(drawable.get_facecolor(), i_norm)
                        new_ec = lighten_color(drawable.get_edgecolor(), i_norm)
                        drawable.set_facecolor(new_fc)
                        drawable.set_edgecolor(new_ec)

        if fig is not None and in_ipython_context:
            ip_display(fig)

    def interact(self, fig=None, axes=None, constraints=None, free_vars=None, caching=True, **kwargs):
        """
        Display an interactive plot where all free variables can be manipulated, with the plot updating accordingly.
        The rest of the variables is considered as dependent.

        :param fig: matplotlib figure to update, can be omitted if axes should be created automatically
        :param axes: matplotlib axes to draw on, can be omitted if axes should be created automatically
        :param constraints: optional sympy (nx1)-matrix of eqns which should be fullfilled (will be "solved" via fmin)
        :param free_vars: optional sympy (rx1)-matrix of symbols which are treated as independent for the constraints
        :param caching: True (default) or False. Determines whether fmin results are cached in a dictionary
        :param kwargs: ipywidgets specifications using the SymPy symbol string representations as keys
                       there are different syntax possibilities:
                       vis_object.interact(x=(xmin, xmax))
                       vis_object.interact(x=(xmin, xmax, step))
                       vis_object.interact(x=(xmin, xmax, step, inistial_value))
        """
        assert in_ipython_context, "Interactive mode only works in an IPython notebook" \
                                   "(maybe you need to install `ipywidgets`, see visualization_requirements.txt)"

        widget_dict = dict()

        if constraints is not None:
            assert isinstance(constraints, sp.MatrixBase)

            solve_constraints = True
            if free_vars is None:
                free_vars = []

            # allow for scalar free var (convenience)
            if isinstance(free_vars, sp.Basic):
                free_vars = [free_vars]

            # distinguish between free and dependent variables
            var_list = list(self.variables)
            free_var_indices = [var_list.index(v) for v in free_vars]
            dependent_var_indices = [var_list.index(v) for v in self.variables if v not in free_vars]
            # dependent_vars = [v for v in self.variables if v not in free_vars]

            n_vars = len(self.variables)

            # expression which will be minimized
            min_expr = constraints.T*constraints
            assert min_expr.shape == (1, 1)

            constraint_norm_func = expr_to_func(self.variables, min_expr[0])

            all_vars = np.zeros((n_vars,))

            def min_target_func(dep_var_vals, free_var_vals):
                """
                Target function for minimization,
                second argument is considered as a parameter
                """

                all_vars[dependent_var_indices] = dep_var_vals
                all_vars[free_var_indices] = free_var_vals

                return constraint_norm_func(*all_vars)

            cbox = Checkbox(value=False, description='solve constraints (fmin)', icon='check',
                            tooltip='solve constraints numerically via fmin')
            widget_dict["chk_solve_constraints"] = cbox

        else:
            solve_constraints = False

        for var in self.variables:
            var_str = repr(var)

            widget_dict[var_str] = self.make_slider_from_kwarg(var_str, kwargs)

        if fig is None or axes is None:
            fig, axes = self.create_default_axes()
            plt.close()

        is_initialized = False
        last_cbox_value = False
        fmin_cache = {}

        # last result for the dependet vars
        last_fmin_result = None

        # noinspection PyShadowingNames
        def interact_fun(**kwargs):
            nonlocal is_initialized
            nonlocal last_cbox_value
            nonlocal last_fmin_result
            widget_var_values = np.array([kwargs[repr(var_symbol)] for var_symbol in self.variables])

            cbox_solve_constraints = kwargs.get("chk_solve_constraints", False)

            print("widget_var_values:", widget_var_values, "cbox:", cbox_solve_constraints)

            if solve_constraints and cbox_solve_constraints:

                free_var_values = [kwargs[repr(var_symbol)] for var_symbol in free_vars]

                # initialize the dep_var_values form widgets if we have no result yet or if the checkbox was unchecked
                if last_fmin_result is None:
                    dep_var_values = widget_var_values[dependent_var_indices]
                    x0_dep_vars = dep_var_values
                else:
                    x0_dep_vars = last_fmin_result

                # dict lookup with the arguments of min_target

                # does not work because we never come to see this key again
                # key_tuple = (tuple(np.round(x0_dep_vars, decimals=5)), tuple(free_var_values))

                key_tuple = tuple(free_var_values)
                cache_content = fmin_cache.get(key_tuple)
                print("cache:", key_tuple, cache_content)
                if caching and cache_content is not None:
                    dep_var_values_result = cache_content
                else:

                    print("calling fmin with x0=", x0_dep_vars, "args=", free_var_values)
                    res = fmin(min_target_func, x0=x0_dep_vars, args=(free_var_values,), full_output=True)
                    dep_var_values_result, fopt, n_it, fcalls, warnflag = res

                    # fill the cache if we had these arguments for the first time (and no error occurred)
                    if caching and warnflag == 0:
                        fmin_cache[key_tuple] = dep_var_values_result

                last_fmin_result = dep_var_values_result

                all_vars[free_var_indices] = free_var_values
                all_vars[dependent_var_indices] = dep_var_values_result

                variables_values = all_vars * 1.0

                # print all coordinates, convert to list for easy copy-pasting (commas)
                print("all coordinates:", list(variables_values))
            else:
                # just use the values from the widgets
                variables_values = widget_var_values

                # reset the cache if checkbox is deactivated
                fmin_cache.clear()
                last_fmin_result = None

            if not is_initialized:
                self.plot_init(variables_values, axes)
                is_initialized = True

            last_cbox_value = cbox_solve_constraints

            self.plot_update(variables_values, axes)
            ip_display(fig)

        # TODO: Maybe return the control elements or something, so that they can be customized
        interact(interact_fun, **widget_dict)

    def add_element(self, points, init_fun, update_fun, **kwargs):
        """
        Add a visualiser element
        :param points: 2x? SymPy matrix or list of 2x1 SymPy vectors describing the defining points as symbolic
        expressions w.r.t the visualisers free variables
        :param init_fun: callable with args (matplotlib axes, 2x? numpy array of points, dict of kwargs) that returns a
        list of matplotlib drawables. Will get called to create all drawables needed by this element.
        :param update_fun: callable with args (matplotlib axes, list of drawables, 2x? numpy array of points, dict of
        kwargs) that returns a list of matplotlib drawables. Will get called every time the plot needs to be updated.
        :param kwargs: arbitrary keyword arguments that get passed to init_fun and update_fun
        """
        if not isinstance(points, sp.Matrix):
            if isinstance(points, list):
                points = st.col_stack(*points)
            else:
                raise Exception("'points' must be a SymPy matrix or a list of column vectors")

        points_fun = st.expr_to_func(self.variables, points, keep_shape=True)
        self.elements.append(VisualiserElement(points_fun, init_fun, update_fun, kwargs))

    def add_linkage(self, points, **kwargs):
        """
        Add a linkage chain element, consisting of round markers at the points and lines connecting them
        :param points: SymPy expressions for the points on the chain
        :param kwargs: keyword arguments passed to matplotlib plot() call
        """
        self.add_element(points, init_linkage, update_linkage, **kwargs)

    def add_polygon(self, points, **kwargs):
        """
        Add a polygon element
        :param points: SymPy expressions for the polygon corners
        :param kwargs: keyword arguments passed to matplotlib Polygon() call
        """
        self.add_element(points, init_polygon, update_polygon, **kwargs)

    def add_disk(self, points, **kwargs):
        """
        Add a disk element, consisting of a circle and a line from the center to the circumference, indicating the
        orientation
        :param points: SymPy expressions for the center point and a point on the circumference
        :param kwargs: keyword arguments passed to matplotlib
        """
        self.add_element(points, init_disk, update_disk, **kwargs)

    @staticmethod
    def make_slider_from_kwarg(key, kwarg_dict):
        """
        Return either a slider or a suitable tuple (from which a slider will be created later).

        :param key:         variable name like "x1"
        :param kwarg_dict:  the whole dictionary which may or may not contain information for the slider widget


        See also: `interact`-method

        :return:
        """

        def fall_back_slider():
            return FloatSlider(min=-5.0, max=5.0, step=0.1, value=0.0)

        if key in kwarg_dict:

            # get the value for that key
            value = kwarg_dict[key]
            warn_msg = "Unkonwn value for kwarg: '{}': {}. See docstring for more information".format(key, value)

            try:
                length = len(value)
            except TypeError:
                warnings.warn(warn_msg, UserWarning)
                return fall_back_slider()

            if length in (2, 3):
                return value
            elif length == 4:
                # this option allows to specify a custom inital value
                return FloatSlider(min=value[0], max=value[1], step=value[2], value=value[3])
            else:
                warnings.warn(warn_msg, UserWarning)
                return fall_back_slider()

        # key was not given. just use the default slider
        return fall_back_slider()


def init_linkage(ax, points, kwargs):
    return ax.plot(points[0, :], points[1, :], **merge_options(kwargs, marker='o', ls='-', lw=3))


def update_linkage(ax, drawables, points, kwargs):
    drawables[0].set_data(points)
    return drawables


def init_polygon(ax, points, kwargs):
    poly = plt.Polygon(points.T, **kwargs)
    ax.add_patch(poly)

    return [poly]


def update_polygon(ax, drawables, points, kwargs):
    poly = drawables[0]
    poly.set_xy(points.T)

    return drawables


def init_disk(ax, points, kwargs):
    kwargs = kwargs.copy()  # don't modify passed in dict
    plot_radius = kwargs.pop("plot_radius", True)

    assert points.shape == (2, 2)
    center_point = points[:, 0]
    border_point = points[:, 1]
    radius = np.sqrt(np.sum((border_point - center_point) ** 2))
    circle = plt.Circle(center_point, radius, **merge_options(kwargs, fill=False))

    line_incompatible_kwargs = ["fill", "edgecolor", "ec", "facecolor", "fc"]
    line_kwargs = {k:v for (k,v) in kwargs.items() if k not in line_incompatible_kwargs}
    line, = ax.plot(points[0, :], points[1, :],
                    **merge_options(line_kwargs,
                        color=circle.get_edgecolor(),
                        ls='-' if plot_radius else 'None'))

    ax.add_patch(circle)

    return [circle, line]


def update_disk(ax, drawables, points, kwargs):
    assert points.shape == (2, 2)
    center_point = points[:, 0]
    border_point = points[:, 1]

    circle = drawables[0]
    # circle.set_center(center_point)
    circle.center = center_point
    radius = np.sqrt(np.sum((border_point - center_point) ** 2))
    circle.set_radius(radius)

    line = drawables[1]
    line.set_data(points[0, :], points[1, :])

    return drawables


class SimAnimation:
    """
    An animation showing the results of some numeric simulation. May consist of multiple kinematic visualisations and
    graphs that get automatically animated based on a sequence of system variable vectors.
    """
    def __init__(self, x_symb, t, x_sim, start_pause=1.0, end_pause=1.0, fig=None, **fig_kwargs):
        """
        Create a new animation, define the system variables, their trajectories and customize the plot figure.
        :param x_symb: symbolic vector of all system variables, matching the columns of x_sim
        :param t: 1D numpy array of equidistant sample times
        :param x_sim: 2D numpy array with the simulation data, rows are time samples, columns are system variables
        :param start_pause: pause in seconds to insert before start of simulation
        :param end_pause: pause in seconds to insert after end of simulation
        :param fig: optional, figure object to be used for the plot, if omitted one will be created
        :param fig_kwargs: keyword arguments to pass to automatically created figure if necessary
        """
        self.x_symb = x_symb
        self.t = t
        self.x_sim = x_sim

        self.n_sim_frames = len(self.t)
        self.dt = (self.t[-1] - self.t[0]) / (self.n_sim_frames - 1)  # assumes equidistant sampling
        self.start_pause_frames = int(start_pause / self.dt)
        self.end_pause_frames = int(end_pause / self.dt)

        # create figure ourselves if none is given
        if fig is None:
            fig = plt.figure(**fig_kwargs)
            plt.close()  # we don't want the empty figure to pop up right now
        self.fig = fig

        # List of tuples, each describing a subplot (either a kinematic visulisation or a graph) in the animation.
        # Each tuple has three entries (matplotlib axes, content, content arguments). If 'content' is a 'Visualiser'
        # object, then 'content arguments' will be a list of column indices in x_sim that correspond to the free
        # variables defined in the 'Visualiser'. If 'content' is a numpy array then the data in that array will be
        # plotted in an animated graph and 'content arguments' contains a dictionary of keyword arguments to be passed
        # to the matplotlib Axes.plot function.
        self.axes = []

        # List of matplotlib drawables that we instantiated in the figure, needed to clear it again.
        self._drawables = []

        # keys are matplotlib Axes objects of the subplots containing graphs, values are lists of Line2D objects of the
        # plot lines; required because they must be updated during animation
        self._graph_lines = {}

        # TODO: clarify this comment
        # The last matplotlib animation object we created, gets reset to None when a subplot gets added. On subsequent
        # calls to display() or something we can return the same animation object which might have the rendered
        # animation already cached, speeding up execution.
        self._cached_anim = None

    def add_visualiser(self, vis, subplot_pos=111, ax=None):
        """
        Add a subplot containing a kinematic visualisation.
        :param vis: Visualiser object
        :param subplot_pos: subplot position, this simply gets passed on to matplotlib's Figure.add_subplot() so every
        valid type of specification should work.
        :param ax: optional, existing Axes object to be used for plotting, if omitted one will be created according to
        the configuration of the Visualiser
        :return: Axes object of the subplot
        """
        assert isinstance(vis, Visualiser)

        if ax is None:
            # use the visualisers configuration for creating a new Axes
            _, ax = vis.create_default_axes(self.fig, subplot_pos)

        # in which columns of x_sim can we find the free variables that the visualiser knows about?
        vis_var_indices = self._find_variable_indices(vis.variables)
        self.axes.append((ax, vis, vis_var_indices))

        # adding a new subplot invalidates any cached animation we might have
        self._cached_anim = None

        return ax

    def add_graph(self, content, subplot_pos=111, ax_kwargs=None, plot_kwargs=None, ax=None):
        """
        Add a subplot containing an animated graph of some system variables.
        :param content: Can be
            - symbolic: SymPy expression, list of SymPy expressions or SymPy matrix, of some system variables or a
                        combination of them
            - numeric: NumPy array of values to be plotted over time, rows are sample times, columns are separate plot
                       lines. The number of rows must match the length of the time vector.
        :param subplot_pos: subplot position, this simply gets passed on to matplotlib's Figure.add_subplot() so every
        valid type of specification should work.
        :param ax_kwargs: keyword arguments to be passed on when creating Axes object
        :param plot_kwargs: keyword arguments to be passed on when calling Axes.plot function
        :param ax: optional, existing Axes object to be used for plotting, if omitted one will be created
        :return: Axes object of the subplot
        """

        # create Axes if necessary
        if ax is None:
            if ax_kwargs is None:
                ax_kwargs = {}
            ax = self.fig.add_subplot(subplot_pos, **ax_kwargs)
            ax.grid()

        assert isinstance(content, np.ndarray) or isinstance(content, sp.Expr) or isinstance(content, sp.Matrix)\
            or isinstance(content, list)

        if isinstance(content, np.ndarray):
            assert content.ndim == 1 or content.ndim == 2, "Data must be one or two-dimensional"
            assert content.shape[0] == self.n_sim_frames, "Data must have as many rows as there are entries in 't' vector"

        # convert all types of symbolic content to a SymPy vector
        if isinstance(content, sp.Expr):
            content = sp.Matrix([content])
        elif isinstance(content, list):
            content = sp.Matrix(content)

        # content is now a SymPy vector or an array of values
        if isinstance(content, np.ndarray):
            # We later expect all data to be two dimensional, so a vector must be converted to a ?x1 matrix
            if content.ndim == 1:
                data = np.reshape(content, (content.shape[0], 1))
            else:
                data = content
        else:
            # content is still symbolic, we need to generate the data vector ourselves
            # instantiate a function that takes one row of x_sim and returns the values to plot at one time instance
            expr_fun = st.expr_to_func(self.x_symb, content, keep_shape=True)

            # allocate memory for the data to plot
            data = np.zeros((self.n_sim_frames, len(content)))

            # use the prepared function to fill the plotting data
            for i in range(data.shape[0]):
                data[i, :] = expr_fun(*self.x_sim[i, :]).flatten()  # expr_fun returns a 2D column vector --> flatten

        if plot_kwargs is None:
            plot_kwargs = {}

        self.axes.append((ax, data, plot_kwargs))

        # adding a new subplot invalidates any cached animation we might have
        self._cached_anim = None

        return ax

    def _anim_init(self):
        """
        Clear the figure and instantiate all drawables that will be updated during animation.
        :return: list of instantiated drawables
        """

        # If anim_init gets called multiple times (as is the case when blit=True), we need to remove
        # all remaining drawables before instantiating new ones
        self.clear_figure()

        for (ax, content, content_args) in self.axes:
            if isinstance(content, Visualiser):
                new_drawables = content.plot_init(np.zeros(len(content.variables)), ax)  # use 0 for all free vars
                self._drawables.extend(new_drawables)
            elif isinstance(content, np.ndarray):
                new_drawables = ax.plot(self.t, content, **content_args)
                self._graph_lines[ax] = new_drawables
                self._drawables.extend(new_drawables)

                handles, labels = ax.get_legend_handles_labels()

                # Create an auto-legend if any plot has defined line labels
                if handles:
                    ax.legend(handles, labels)

        return self._drawables

    def _anim_update(self, i):
        """
        Update the figure elements with the data in frame with index i
        :param i: frame index in t and x_sim vector
        :return: list of updated drawables
        """
        drawables = []

        for (ax, content, content_args) in self.axes:
            if isinstance(content, Visualiser):
                vis_var_indices = content_args
                # extract the free variable values in the right order (the one the Visualiser expects)
                vis_var_values = self.x_sim[i, vis_var_indices]

                drawables.extend(content.plot_update(vis_var_values, ax))
            elif isinstance(content, np.ndarray):
                # the line objects in this subplot that we created earlier
                lines = self._graph_lines[ax]

                for line_i, line in enumerate(lines):
                    # update each line with the data up to (and including) the current time
                    line.set_data(self.t[:i + 1], content[:i + 1, line_i])

                drawables.extend(lines)

        return drawables

    def plot_frame(self, frame_number=None):
        """
        Plot a single frame onto the figure.
        :param frame_number: optional, index of frame to plot, defaults to the last frame
        :return: the figure object that was plotted onto
        """
        if frame_number is None:  # Default to the last frame
            frame_number = self.n_sim_frames - 1

        assert 0 <= frame_number < self.n_sim_frames,\
            f"Frame number needs to be in the supplied data range [0, {self.n_sim_frames-1}]"

        self._anim_init()
        self._anim_update(frame_number)

        return self.fig

    def display_frame(self, frame_number=None):
        """
        Display the plot of a single frame.
        :param frame_number: optional, index of frame to plot, defaults to the last frame
        :return:
        """
        assert in_ipython_context, "Display only works in an IPython notebook"
        fig = self.plot_frame(frame_number)
        ip_display(fig)

    def to_animation(self):
        """
        Convert to a matplotlib Animation object.
        :return: matplotlib Animation object
        """

        # if we have something cached, we don't need to create a new one
        if self._cached_anim:
            return self._cached_anim

        # Transform frame index before calling anim_update. All indices < start_pause_frames will result in anim_update
        # being called with i=0, all indices > start_pause_frames + n_sim_frames will result in anim_update being called
        # with i=n_sim_frames-1. This effectively creates pauses of start_pause_frames and end_pause_frames length
        # respectively.
        def anim_update_with_pause(i):
            self._anim_update(max(min(i - self.start_pause_frames, self.n_sim_frames - 1), 0))

        anim = animation.FuncAnimation(self.fig, anim_update_with_pause, init_func=self._anim_init,
                                       frames=self.n_sim_frames + self.start_pause_frames + self.end_pause_frames,
                                       interval=1000 * self.dt)

        self._cached_anim = anim

        return anim

    def display(self, with_js=True):
        """
        Display the animation.
        :param with_js: should the displayed element have fancy playback controls

        See also: display_video_file (function)
        """
        assert in_ipython_context, "Display only works in an IPython notebook"

        if with_js:
            html_source = self.to_animation().to_jshtml()
        else:
            html_source = self.to_animation().to_html5_video()

        # noinspection PyTypeChecker
        ip_display(HTML(html_source))

    def save(self, file_name, **kwargs):
        """
        Save the animation.
        :param file_name: path of destination file
        :param kwargs: keyword arguments to passed to matplotlib's Animation.save() function
        """
        anim = self.to_animation()
        anim.save(file_name, **merge_options(kwargs, writer='imagemagick' if file_name.endswith('gif') else None))

    def clear_figure(self, reset_color_cycle=True):
        """
        Remove all drawables we created from the figure.
        :param reset_color_cycle: should the cycle of automatic plot colors also be reset
        """
        while self._drawables:
            drawable = self._drawables.pop()
            drawable.remove()

        if reset_color_cycle:
            for (ax, _, _) in self.axes:
                ax.set_prop_cycle(None)

    def _find_variable_indices(self, variables):
        assert all([var in self.x_symb for var in variables])

        indices = np.zeros(len(variables), dtype=int)
        for var_i, var in enumerate(variables):
            for x_i in range(len(self.x_symb)):
                if self.x_symb[x_i] == var:
                    indices[var_i] = x_i
                    break

        return indices


def display_video_file(fname, width=None, height=None):
    """
    Display a video file
    :param fname:   filename
    :param width:   desired width of the video (optional)
    :param height:  desired height of the video (optional)
    """
    assert in_ipython_context, "Display only works in an IPython notebook"

    if isinstance(width, int):
        width_str = "width={}".format(width)
    else:
        width_str=""

    if isinstance(height, int):
        height_str = "height={}".format(height)
    else:
        height_str=""

    html_source = """
    <video {} {} controls loop="loop">
        <source src="{}" type="video/mp4">
    </video>
    """.format(width_str, height_str, fname)

    # noinspection PyTypeChecker
    ip_display(HTML(html_source))
