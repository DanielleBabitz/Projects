import plotly.express as px
import ipywidgets as widgets
from IPython.display import display
import plotly.subplots as sp
import plotly.graph_objects as go

#utility functions for interactive plotting:

def create_feature_dropdown(features):
    """Create a dropdown widget for selecting features"""
    return widgets.Dropdown(
        options=features,
        description='Feature:',
    )

def create_filter_widgets(data, filters):
    """Create filtering widgets"""
    filter_widgets = {}
    for filter_name in filters:
        options = data[filter_name].unique()

        checkbox = widgets.Checkbox(
            value=True,
            description=f'Select All {filter_name}s'
        )

        selector = widgets.SelectMultiple(
            options=options,
            value=options.tolist(),  # default to all
            description=f'{filter_name}s:',
            disabled=False,
            style={'description_width': 'initial'}
        )

        def update_selection(change, name=filter_name, sel=selector):
            sel.value = options.tolist() if checkbox.value else []

        checkbox.observe(update_selection, names='value')
        filter_widgets[filter_name] = (checkbox, selector)

    return filter_widgets

#default function for figure layout:
def set_figure_layout(fig, title, xaxis_title, yaxis_title, width=800, height=600, show_colorbar=False):
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=width,
        height=height,
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        title_font=dict(size=20, color='black'),
        xaxis_title_font=dict(size=14, color='black'),
        yaxis_title_font=dict(size=14, color='black'),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
    )
    if show_colorbar:
        fig.update_coloraxes(colorbar=dict(thickness=15, title='', titleside='right'))

#Plotting functions
#   DATA: pandas dataframe
#   FEATURES: the names of the columns to plot
#   FILTERS: the name of the columns to apply filtering by (unique values)


def plot_distributions(data, columns):
    """Plot distributions of all chosen variables, 3 in a row, not interactive"""
    num_vars = len(columns)
    rows = (num_vars // 3) + (num_vars % 3 > 0)

    fig = sp.make_subplots(rows=rows, cols=3, subplot_titles=[x[:40] for x in columns])
    for i, column in enumerate(columns):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.add_trace(
            go.Histogram(x=data[column], name=column, opacity=0.75),
            row=row, col=col
        )

    fig.update_layout(height=300 * rows, width=1200, title_text='Distributions of Variables', showlegend=False)
    fig.show()

def plot_interactive_distributions(data, features, filters):
    """
    Plot an interactive histogram with optional filtering
    the filters have to be a column in the dataframe
    """
    feature_dropdown = create_feature_dropdown(features)
    filter_widgets = create_filter_widgets(data, filters)

    def update_histogram(feature, **filter_values):
        filtered_df = data
        for filter_name, values in filter_values.items():
            if values:
                filtered_df = filtered_df[filtered_df[filter_name].isin(values)]

        if filtered_df.empty:
            print("No data available for the selected filters.")
            return

        fig = px.histogram(filtered_df, x=feature, histnorm='percent', title=f'Distribution of {feature}',
                           marginal='violin', color_discrete_sequence=["#52B788"])
        set_figure_layout(fig, f'Distribution of {feature}', feature, 'Percentage')
        fig.show()

    output_dict = {'feature': feature_dropdown}
    for filter_name, (checkbox, selector) in filter_widgets.items():
        output_dict[filter_name] = selector

    output = widgets.interactive_output(update_histogram, output_dict)
    controls = widgets.VBox([widgets.HBox([checkbox, selector]) for checkbox, selector in filter_widgets.values()] + [feature_dropdown])
    display(controls, output)

def plot_map_with_slider(data, country_label, features, time_filter):
    """Plot a world map with a feature selection and a time slider"""
    feature_dropdown = create_feature_dropdown(features)

    filter_slider = widgets.SelectionSlider(
        options=sorted(data[time_filter].unique()),
        value=data[time_filter].min(),  # Default to the minimum year
        description=f'{time_filter}:',
        continuous_update=False,
        layout={'width': '1000px'},
        style={'description_width': 'initial'}
    )

    def update_map(feature, selected_time):
        filtered_df = data[data[time_filter] == selected_time]
        if filtered_df.empty:
            print("No data available for the selected time filter.")
            return

        fig = px.choropleth(
            filtered_df,
            locations=country_label,
            locationmode='country names',
            color=feature,
            hover_name=country_label,
            color_continuous_scale=px.colors.sequential.YlOrRd,
            labels={feature: feature},
            title=f'{feature} in {selected_time}'
        )

        fig.update_geos(projection_type="natural earth")
        set_figure_layout(fig, f'{feature} in {selected_time}', 'Country', feature, show_colorbar=True)

        fig.show()

    output_map = widgets.interactive_output(update_map, {'feature': feature_dropdown, 'selected_time': filter_slider})
    display(feature_dropdown, filter_slider, output_map)

def plot_lines_interactive(data, x, features, filters=None, enable_filter=True, agg='mean'):
    """
    Plot interactive line graphs with optional filtering
    :param agg: function to aggregate the data by, default mean
    """
    if filters is None:
        filters = []

    feature_dropdown = create_feature_dropdown(features)
    filter_widgets = create_filter_widgets(data, filters)

    # Include "All" option for overall mean
    all_option = "All"
    filter_selector = filter_widgets[filters[0]][1]  # Grab the selector
    filter_selector.options = [all_option] + list(filter_selector.options)

    def update_plot(feature, filter_values):
        filtered_df = data.copy()
        if filter_values and all_option not in filter_values:
            filtered_df = filtered_df[filtered_df[filters[0]].isin(filter_values)]

        if filtered_df.empty:
            print("No data available for the selected filters.")
            return

        fig = go.Figure()
        for value in filter_values:
            if value == all_option:
                overall_mean_df = data.groupby(x, as_index=False)[feature].mean()
                fig.add_trace(go.Scatter(
                    x=overall_mean_df[x],
                    y=overall_mean_df[feature],
                    mode='lines+markers',
                    name=f'Overall {agg}',
                    marker=dict(size=8, color='red'),  # Different color for overall mean
                    line=dict(width=2),  # Solid line for overall mean
                    opacity=0.8
                ))
            else:
                mean_value_df = filtered_df[filtered_df[filters[0]] == value].groupby(x, as_index=False)[feature].agg(agg)
                fig.add_trace(go.Scatter(
                    x=mean_value_df[x],
                    y=mean_value_df[feature],
                    mode='lines+markers',
                    name=value,
                    marker=dict(size=8),
                    line=dict(width=2),
                    opacity=0.8
                ))

        set_figure_layout(fig, f'{feature} Over Time', x, feature)

        fig.show()

    output = widgets.interactive_output(update_plot, {'feature': feature_dropdown, 'filter_values': filter_selector})
    controls = widgets.VBox([feature_dropdown, filter_selector])
    display(controls, output)


def plot_scatter_interactive(data, x_col, y_col, hover_data=None, color_filter=None, selector_filter=None, regression_line=False):
    """
    Create an interactive scatter plot with options for filtering

    :param data: DataFrame containing the data
    :param x_col: Column name for x-axis
    :param y_col: Column name for y-axis
    :param hover_data (optional): Column name(s) to show on hover (e.g., 'Entity')
    :param color_filter (optional): Column name of a filter to be applied by color (and can be selected via legend)
    :param selector_filter (optional): Column name of a filter to be applied by a selector

    """
    filter = False
    if selector_filter:
        filter = True
        filter_widgets = create_filter_widgets(data, [selector_filter])

    def update_scatter(**filter_values):
        filtered_df = data

        if filter:
            for filter_name, values in filter_values.items():
                if values:
                    filtered_df = filtered_df[filtered_df[filter_name].isin(values)]

            if filtered_df.empty:
                print("No data available for the selected filters.")
                return

        # Create scatter plot
        fig = px.scatter(
            filtered_df,
            x=x_col,
            y=y_col,
            color=color_filter,
            hover_name=hover_data,
            size_max=60,
            opacity=0.6,
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={x_col: x_col, y_col: y_col},
            trendline='ols' if regression_line else None
        )

        # Update layout
        fig.update_layout(
            xaxis=dict(
                title=x_col,
                gridcolor='LightGray',
                showgrid=True
            ),
            yaxis=dict(
                title=y_col,
                gridcolor='LightGray'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            width=800,
            height=600
        )

        fig.show()

    # Create interactive output
    if filter:
        output = widgets.interactive_output(
            update_scatter,
            {name: widget[1] for name, widget in filter_widgets.items()}
        )
        controls = widgets.VBox(
            [widget[0] for widget in filter_widgets.values()] + [widget[1] for widget in filter_widgets.values()])
        display(controls, output)

    else:
        update_scatter()
