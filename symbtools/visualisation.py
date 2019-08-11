import numpy as np
import sympy as sp
import symbtools as st
from ipywidgets import FloatSlider, interact
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display


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
            fig = plt.figure()  # TODO: Get rid of pyplot calls, we don't want to affect the global state machine
        ax = fig.add_subplot(add_subplot_args, **merge_options(self.axes_kwargs, aspect='equal', xlim=(-2.0, 2.0), ylim=(-2.0, 2.0)))
        ax.grid()

        return fig, ax

    def plot(self, variables_values, axes=None):
        """
        Plot for some specific variable values.
        :param variables_values: iterable of values for all free variables
        :param axes: the matplotlib axes to plot on, one will be created if none is given
        """
        assert len(self.variables) == len(
            variables_values), f"You need to pass as many variable values as this visualiser has variables. Required: {len(self.variables)}, Given: {len(variables_values)}"

        fig = None
        if axes is None:
            fig, axes = self.create_default_axes()
            plt.close()  # TODO: Get rid of pyplot calls, we don't want to affect the global state machine

        self.plot_init(variables_values, axes)
        self.plot_update(variables_values, axes)
        if fig is not None:
            display(fig)  # TODO: Don't just call display, we might not be in an IPython context

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

    def interact(self, fig=None, axes=None, **kwargs):
        """
        Display an interactive plot where all free variables can be manipulated, with the plot updating accordingly
        :param fig: matplotlib figure to update, can be ommitted if axes should be created automatically
        :param axes: matplotlib axes to draw on, can be ommitted if axes should be created automatically
        :param kwargs: ipywidgets specifications using the SymPy symbol string representations as keys
        """
        widget_dict = dict()

        for var in self.variables:
            var_str = repr(var)
            if var_str in kwargs:
                widget_dict[var_str] = kwargs[var_str]
            else:
                widget_dict[var_str] = FloatSlider(min=-5.0, max=5.0, step=0.1, value=0.0)

        if fig is None or axes is None:
            fig, axes = self.create_default_axes()
            plt.close()

        is_initialized = False

        def interact_fun(**kwargs):
            nonlocal is_initialized
            variables_values = [kwargs[repr(var_symbol)] for var_symbol in self.variables]

            if not is_initialized:
                self.plot_init(variables_values, axes)
                is_initialized = True

            self.plot_update(variables_values, axes)
            display(fig)

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
    assert points.shape == (2, 2)
    center_point = points[:, 0]
    border_point = points[:, 1]
    radius = np.sqrt(np.sum((border_point - center_point) ** 2))
    circle = plt.Circle(center_point, radius, **merge_options(kwargs, fill=False))
    line, = ax.plot(points[0, :], points[1, :], **merge_options(kwargs, color=circle.get_edgecolor()))

    ax.add_patch(circle)

    return [circle, line]


def update_disk(ax, drawables, points, kwargs):
    assert points.shape == (2, 2)
    center_point = points[:, 0]
    border_point = points[:, 1]

    circle = drawables[0]
    circle.set_center(center_point)
    radius = np.sqrt(np.sum((border_point - center_point) ** 2))
    circle.set_radius(radius)

    line = drawables[1]
    line.set_data(points[0, :], points[1, :])

    return drawables


class SimAnimation:
    def __init__(self, x_symb, t, x_sim, fig=None, **fig_kwargs):
        self.x_symb = x_symb
        self.t = t
        self.x_sim = x_sim
        if fig is None:
            fig = plt.figure(**fig_kwargs)
            plt.close()
        self.fig = fig
        self.axes = []

    def add_visualiser(self, vis, subplot_pos=111, ax=None):
        if ax is None:
            _, ax = vis.create_default_axes(self.fig, subplot_pos)

        assert isinstance(vis, Visualiser)
        vis_var_indices = self._find_variable_indices(vis.variables)
        self.axes.append((ax, vis, vis_var_indices))

        return ax

    def add_graph(self, expr, subplot_pos=111, ax_kwargs=None, plot_kwargs=None, ax=None):
        if ax is None:
            if ax_kwargs is None:
                ax_kwargs = dict()
            ax = self.fig.add_subplot(subplot_pos, **ax_kwargs)
            ax.grid()
        assert isinstance(expr, sp.Expr) or isinstance(expr, sp.Matrix) or isinstance(expr, list)

        if isinstance(expr, sp.Expr):
            expr = sp.Matrix([expr])
        elif isinstance(expr, list):
            expr = sp.Matrix(expr)

        expr_fun = st.expr_to_func(self.x_symb, expr, keep_shape=True)
        data = np.zeros((len(self.t), len(expr)))

        for i in range(data.shape[0]):
            data[i, :] = expr_fun(*self.x_sim[i, :]).flatten()

        if plot_kwargs is None:
            plot_kwargs = dict()

        self.axes.append((ax, data, plot_kwargs))

        return ax

    def display(self):
        init_drawables = []
        graph_lines = {}

        def anim_init():
            nonlocal init_drawables

            # If anim_init gets called multiple times (as is the case when blit=True), we need to remove
            # all remaining drawables before instantiating new ones
            while init_drawables:
                drawable = init_drawables.pop()
                drawable.remove()

            for (ax, content, content_args) in self.axes:
                if isinstance(content, Visualiser):
                    new_drawables = content.plot_init(np.zeros(len(content.variables)), ax)
                    init_drawables += new_drawables
                elif isinstance(content, np.ndarray):
                    new_drawables = ax.plot(self.t, content, **content_args)
                    graph_lines[ax] = new_drawables
                    init_drawables += new_drawables

                    handles, labels = ax.get_legend_handles_labels()

                    # Create an auto-legend if any plot has defined line labels
                    if handles:
                        ax.legend(handles, labels)

            return init_drawables

        def anim_update(i):
            drawables = []

            for (ax, content, content_args) in self.axes:
                if isinstance(content, Visualiser):
                    vis_var_indices = content_args
                    vis_var_values = self.x_sim[i, vis_var_indices]

                    drawables += content.plot_update(vis_var_values, ax)
                elif isinstance(content, np.ndarray):
                    lines = graph_lines[ax]

                    for line_i, line in enumerate(lines):
                        line.set_data(self.t[:i+1], content[:i+1, line_i])

                    drawables += lines

            return drawables

        anim = animation.FuncAnimation(self.fig, anim_update, init_func=anim_init, frames=len(self.t),
                                       interval=1000 * (self.t[-1] - self.t[0]) / (len(self.t) - 1))
        display(HTML(anim.to_jshtml()))

    def _find_variable_indices(self, variables):
        assert all([var in self.x_symb for var in variables])

        indices = np.zeros(len(variables), dtype=int)
        for var_i, var in enumerate(variables):
            for x_i in range(len(self.x_symb)):
                if self.x_symb[x_i] == var:
                    indices[var_i] = x_i
                    break

        return indices