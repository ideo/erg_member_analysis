import datetime
import json
import numpy as np
import pandas as pd

import streamlit as st
import altair as alt
# from altair.utils.data import to_values

import pydeck as pdk

# from timezonefinder import TimezoneFinder


ideo_employees_march_2023_file = 'data/IDEO_Airtable_Employee_data.json'
horizon_members_march_2023_file = "data/horizon_members_2023_03.txt"
black_design_member_march_2023_file = "data/black_design_members_2023_03.txt"

power = ['Individual', 'Team', 'Director', 'Enterprise', ]
management_levels = [
    'Individual',
    'Senior Individual',
    'Team',
    'Senior Team',
    'Director',
    'Senior Director',
    'Enterprise',
    'Senior Enterprise',
]
internal_cost_centers = ['Facilities', 'Experience', 'Talent', 'Enterprise', 'Legal', 'Technology',
                         'Marketing', 'Finance', 'BD', 'Global']

# https://gps-coordinates.org/
studio_names = {
    'Cambridge': {'lat': 42.3668233, 'long': -71.1060706},
    'Chicago': {'lat': 41.883718, 'long': -87.632382},
    'San Francisco': {'lat': 37.73288916682891, 'long': -122.5024402141571},
    'London': {'lat': 51.5033466, 'long': -0.0793965},
    'Munich': {'lat': 48.1379879, 'long': 11.575182},
    'Shanghai': {'lat': 31.2203102, 'long': 121.4623931},
    'Singapore': {'lat': 1.351616, 'long': 103.808053},
    'Tokyo': {'lat': 35.689506, 'long': 139.6917},
}
identifiers = ['Employee_ID', 'Worker', 'Email_-_Work', 'Preferred_Name', ]
exclude_from_plots = ['Hire_Date', 'Cost_Center', 'Region', ]

biz_details = [
    # 'businessTitle', 'Position', # businessTitle == Position?? - nope some diff
    'Active_Status',
    'Cost_Center',
    # 'Job_Profile',
    'Job_Family',
    # 'Craft_Cohort',
    # 'Domain',
    'On_Leave',
    'Management_Level',
    'Time_Type', 'Worker_Type',
    'Hire_Date',
    'Region',
    'location']

general_info = ['Job_Family', 'Time_Type', 'Worker_Type']
level_section = ['Management_Level', 'level_group', 'tenure_in_yrs', 'cost_center_type']
location_section = ['location', 'studio', 'Region', 'region_simplified']


def load_all_employee_data():
    with open(ideo_employees_march_2023_file, encoding='utf-8') as file:
        j = json.load(file)

    employee_list = list(j.values())[0]  # list of dicts
    df = pd.DataFrame(employee_list)

    df = df[df['Active_Status'] == '1'].copy()

    sub_cols = []
    sub_cols.extend(identifiers)
    sub_cols.extend(biz_details)
    return df[sub_cols].copy()


def add_level_groups(df):
    # 2 enterprise individuals NA - this is rough fix
    df['Management_Level'].fillna('Senior Enterprise', inplace=True)

    df['level_group'] = df['Management_Level']

    df.loc[df['Management_Level'].str.contains('Individual', na=False), 'level_group'] = 'Individual'
    df.loc[df['Management_Level'].str.contains('Team', na=False), 'level_group'] = 'Team'
    df.loc[df['Management_Level'].str.contains('Director', na=False), 'level_group'] = 'Director'
    df.loc[df['Management_Level'].str.contains('Enterprise', na=False), 'level_group'] = 'Enterprise'

    return df


def add_cost_center_type(df):
    df['cost_center_type'] = df['Cost_Center']

    for job_family in internal_cost_centers:
        df.loc[df['Cost_Center'].str.contains(job_family), 'cost_center_type'] = 'Internal'

    df.loc[df['Cost_Center'].str.contains('General'), 'cost_center_type'] = 'External'
    df.loc[df['Cost_Center'].str.contains('IDEO U'), 'cost_center_type'] = 'External'
    df.loc[df['Cost_Center'].str.contains('Open Financial Systems'), 'cost_center_type'] = 'External'
    df.loc[df['Cost_Center'].str.contains('Shop'), 'cost_center_type'] = 'External'
    df.loc[df['Cost_Center'].str.contains('Production'), 'cost_center_type'] = 'External'
    df.loc[df['Cost_Center'].str.contains('Creative Leadership'), 'cost_center_type'] = 'External'

    return df


def add_true_regions(df):
    true_region_mapping = {
        'NA': ['Remote', 'Lion', 'Cambridge', 'Chicago', 'San Francisco'],
        'Asia': ['Tokyo', 'Shanghai', 'Singapore'],
        'Europe': ['Munich', 'London'],
    }
    for k, v in true_region_mapping.items():
        df.loc[df['location'].str.contains('|'.join(v)), 'region_simplified'] = k

    return df


def clean_studio_names(df):
    df['studio'] = df['location']
    df.loc[df['location'].str.contains('Remote|Cloud'), 'studio'] = 'Cloud'
    for studio in studio_names:
        df.loc[df['location'].str.contains(studio), 'studio'] = studio
    return df


def clean_geographic_data(df):
    df = clean_studio_names(df)
    df = add_true_regions(df)

    return df


def add_ideo_tenure(df):
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'])
    df['tenure_in_yrs'] = (datetime.datetime.now() - df['Hire_Date']) / np.timedelta64(1, 'Y')
    return df


def check_for_non_ideo_com_members(member_emails, employee_df):
    ideo_email_list = employee_df['Email_-_Work'].unique().tolist()
    outside_ideo_com = list(set(member_emails) - set(ideo_email_list))
    if outside_ideo_com:
        if outside_ideo_com != [""]:
            email_cnt = len(outside_ideo_com)
        else:
            email_cnt = 0
        st.header(f'ERG members without ideo.com email address OR not in Workday data: {email_cnt}')
        my_expander = st.expander(label='Expand me to see emails')
        with my_expander:
            st.write(outside_ideo_com)


def load_data(member_emails):
    employee_data_df = load_all_employee_data()
    employee_data_df = clean_geographic_data(employee_data_df)
    add_ideo_tenure(employee_data_df)

    erg_member_data_df = employee_data_df[employee_data_df['Email_-_Work'].isin(member_emails)].copy()
    erg_member_data_df.reset_index(inplace=True, drop=True)
    erg_member_data_df = add_level_groups(erg_member_data_df)
    erg_member_data_df = add_cost_center_type(erg_member_data_df)

    check_for_non_ideo_com_members(member_emails, erg_member_data_df)

    return erg_member_data_df


def fill_chart(df, x, y, xbin=False, ysort=None, tooltip=None):
    if tooltip:
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                alt.X(x, bin=xbin),
                alt.Y(y, sort=ysort),
                alt.Color("level_group", sort=power, scale=alt.Scale(scheme='magma')),
                tooltip=tooltip
            )
            .interactive()
        )
    else:
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                alt.X(x, bin=xbin),
                alt.Y(y, sort=ysort),
                alt.Color("level_group", sort=power, scale=alt.Scale(scheme='magma')),
            )
            .interactive()
        )


def plot_general_info(erg_df):
    st.title('General Employee Data')
    st.caption('Source: Workday March 2023 - Contains some errors')
    st.subheader(f'Raw IDEO.com Employee Data: {erg_df.shape[0]}')
    st.dataframe(erg_df)

    col1, col2 = st.columns([3, 2])
    streamlit_cols = [col1, col2, col2]

    for i, col in enumerate(general_info):
        with streamlit_cols[i]:
            # COUNTS BY CATEGORY
            group_sizes = erg_member_df.groupby(col).size().reset_index(name='count')
            # if group_sizes.shape[0] > 1:
            # TITLE & DATA
            st.subheader(col)
            my_expander = st.expander(label='Expand me')
            with my_expander:
                st.dataframe(group_sizes)
            # PLOT
            x = "count()"
            y = f"{col}:O"
            ysort = '-x'
            tooltip = ["Worker", "cost_center_type", "Cost_Center", "Management_Level",
                       alt.Tooltip('tenure_in_yrs:Q', format=",.2f"), "location"]
            chart = fill_chart(erg_df, x=x, y=y, ysort=ysort, tooltip=tooltip)
            st.altair_chart(chart)


def plot_ridge_line(df):
    # from vega_datasets import data

    # source = data.seattle_weather.url
    step = 20
    overlap = 1

    chart = alt.Chart(df, height=step).transform_timeunit(
        Month='month(date)'
    ).transform_joinaggregate(
        mean_temp='mean(temp_max)', groupby=['Month']
    ).transform_bin(
        ['bin_max', 'bin_min'], 'temp_max'
    ).transform_aggregate(
        value='count()', groupby=['Month', 'mean_temp', 'bin_min', 'bin_max']
    ).transform_impute(
        impute='value', groupby=['Month', 'mean_temp'], key='bin_min', value=0
    ).mark_area(
        interpolate='monotone',
        fillOpacity=0.8,
        stroke='lightgray',
        strokeWidth=0.5
    ).encode(
        alt.X('bin_min:Q', bin='binned', title='Maximum Daily Temperature (C)'),
        alt.Y(
            'value:Q',
            scale=alt.Scale(range=[step, -step * overlap]),
            axis=None
        ),
        alt.Fill(
            'mean_temp:Q',
            legend=None,
            scale=alt.Scale(domain=[30, 5], scheme='redyellowblue')
        )
    ).facet(
        row=alt.Row(
            'Month:T',
            title=None,
            header=alt.Header(labelAngle=0, labelAlign='right', format='%B')
        )
    ).properties(
        title='Seattle Weather',
        bounds='flush'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    ).configure_title(
        anchor='end'
    )
    st.altair_chart(chart)


def remove_contingency_option(erg_df, section):
    types = erg_df['Worker_Type'].unique()
    df = erg_df.copy()
    if "Contingent Worker" in types:
        exclude_contingency = st.radio(
            "Exclude Contingency Workers?",
            ('No', 'Yes'),
            key=section
        )

        if exclude_contingency == "Yes":
            df = erg_df[erg_df['Worker_Type'] != 'Contingent Worker'].copy()
    return df


def plot_level_info(erg_df):
    st.title('Power distribution')
    df = remove_contingency_option(erg_df, section="level")

    for col in level_section:
        col1, col2 = st.columns([3, 2])
        with col1:
            if col == 'tenure_in_yrs':
                x = f"{col}:Q"
                y = "count()"
                chart = fill_chart(df, x=x, y=y, xbin=True)
                # plot_ridge_line(erg_df)
            else:
                if col == 'level_group':
                    x = "count()"
                    y = f"{col}:O"
                    # chart = fill_chart(erg_df, x=x, y=y, ysort='-x')
                    chart = fill_chart(df, x=x, y=y, ysort=management_levels)

                else:
                    x = "count()"
                    y = f"{col}:O"
                    tooltip = ["Worker", "cost_center_type", "Cost_Center", "Management_Level",
                               alt.Tooltip('tenure_in_yrs:Q', format=",.2f"), "location"]
                    # chart = fill_chart(erg_df, x=x, y=y, ysort='-x', tooltip=tooltip)
                    chart = fill_chart(df, x=x, y=y, ysort=management_levels, tooltip=tooltip)

            st.subheader(col)
            st.altair_chart(chart)
        with col2:
            if col == 'tenure_in_yrs':
                erg_df[col] = df[col].round()

            group_sizes = df.groupby(col).size().reset_index(name='count')
            st.dataframe(group_sizes)


def plot_location_info(erg_df):
    st.title("Where is everyone?")
    df = remove_contingency_option(erg_df, section="location")

    for col in location_section:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader(col)
            x = "count()"
            y = f"{col}:O"
            ysort = '-x'
            chart = fill_chart(df, x=x, y=y, ysort=ysort)
            st.altair_chart(chart)
        with col2:
            group_sizes = df.groupby(col).size().reset_index(name='count')
            st.dataframe(group_sizes)


def plot_data(erg_df):
    plot_general_info(erg_df)
    plot_level_info(erg_df)
    plot_location_info(erg_df)


def geo_mapping(df):
    # GET GEO DATA
    geo_data = pd.read_csv('data/us_states.tsv', sep='\t')
    df.loc[df['region_simplified'].str.contains('NA'), 'state'] = \
        df.location.apply(lambda x: x.split(' ')[1])
    # COMBINE LAT LONG
    df = pd.merge(df, geo_data, on='state', how='left')
    # ASSIGN LAT LONG FOR NON NA
    for name in studio_names:
        if name == 'Cloud':
            pass
        else:
            df.loc[df['studio'].str.contains(name), 'latitude'] = studio_names[name]['lat']
            df.loc[df['studio'].str.contains(name), 'longitude'] = studio_names[name]['long']

    # ADD TIMEZONE DATA
    # tf = TimezoneFinder()  # reuse
    # erg_member_df['timezone'] = erg_member_df.apply(lambda x: tf.timezone_at(lng=x.longitude, lat=x.latitude), axis=1)

    st.write(erg_member_df)
    # st.write(geo_data)
    st.map(erg_member_df)

    groups = erg_member_df.groupby(['latitude', 'longitude']).size().reset_index(name='count')
    groups["count_scaled"] = groups["count"].apply(lambda count: count * 10000)

    # map default to the center of the data
    # midpoint = (np.average(erg_member_df['latitude']), np.average(erg_member_df['longitude']))
    # Set the viewport location
    # view_state = pdk.ViewState(
    #     latitude=midpoint[0],
    #     longitude=midpoint[1],
    #     # Magnification level of the map, usually between 0 (representing the whole world) and 24 (close to
    #     # individual buildings)
    #     zoom=-5
    # )

    # pdk.data_utils.viewport_helpers.compute_view(groups[['longitude', 'latitude']])
    # Define a layer to display on a map
    layer = pdk.Layer(
        "ScatterplotLayer",
        # erg_member_df,
        groups,
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=['longitude', 'latitude'],
        get_radius="count_scaled",
        # get_radius=500,
        get_fill_color=[255, 140, 0],
        get_line_color=[0, 0, 0],
    )

    # # Set the viewport location
    # view_state = pdk.ViewState(
    #     # latitude=37.7749295, longitude=-122.4194155,
    #     zoom=5,
    #     # bearing=500,
    #     # pitch=500
    # )

    # Render
    st.pydeck_chart(pdk.Deck(layers=[layer],
                             # initial_view_state=view_state
                             ))
    # r.to_html("screengrid_layer.html")

    # # Define a layer to display on a map
    # layer = pdk.Layer(
    #     type='ScatterplotLayer',
    #     data=erg_member_df,
    #     radiusScale=250,
    #     radiusMinPixels=5,
    #     getFillColor=[248, 24, 148],
    # )
    #
    # st.pydeck_chart(pdk.Deck(
    #     layers=[layer],
    #     initial_view_state=view_state,
    # )
    # )


if __name__ == '__main__':

    dataset = st.radio(
        "Which ERG members?",
        ('Horizon - March 2023', 'Black x Design - March 2023', 'load my own'))

    if dataset == 'load my own':
        raw_emails = st.text_area('Emails from slack channel', '''''')
        erg_member_emails = raw_emails.split(", ")
        if raw_emails:
            member_cnt = len(erg_member_emails)
        else:
            member_cnt = 0
    else:
        erg_email_file = None
        if dataset == 'Horizon - March 2023':
            erg_email_file = horizon_members_march_2023_file
        elif dataset == 'Black x Design - March 2023':
            erg_email_file = black_design_member_march_2023_file

        with open(erg_email_file, encoding='utf8') as f:
            erg_member_emails = f.read().splitlines()
            member_cnt = len(erg_member_emails)

    st.caption(f'Number of member emails loaded: {member_cnt}')
    erg_member_df = load_data(erg_member_emails)
    plot_data(erg_member_df)

    # MAPPING - MESSY CODE / NOT SO USEFUL YET
    # geo_mapping(erg_member_df)
