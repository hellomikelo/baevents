import streamlit as st
from streamlit_gsheets import GSheetsConnection
from datetime import datetime
import pandas as pd
from ast import literal_eval
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.set_page_config(layout="wide")

# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read()
# df = conn.read(
#     worksheet="Sheet1",
#     ttl="10m",
#     usecols=[0, 1],
#     nrows=3,
# )

# Preprocss df before 
def clean_df(df):
    # df['Tags'] = df['Tags'].apply(lambda x: [] if x == '' else literal_eval(x))
    df['Categories'] = df['Categories'].apply(pd.eval)
    # df['Loc Coord'] = df['Loc Coord'].apply(lambda x: eval(x))
    df['Zip Code'] = df['Zip Code'].astype('Int64').astype('str')
    # df = df.drop(columns=['Loc Coord'])
    return df

df = clean_df(df)

st.title("📚 Bay Area Library Events")

st.write(
    """Browse free public library events around the Bay Area in an easily searchable table. Use `Filters` to narrow down the list.
    The table refreshes every day at midnight.
    """
)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    # modify = st.checkbox("`Filters`")

    # if not modify:
    #     return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter table on", df.columns, default=['Description'])
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if column in ['City'] or is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    # default=list(df[column].unique()),
                    default=[],
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f'Search substring or regex in column "{column}"',
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input, case=False, regex=True)]

    return df


st.dataframe(
    filter_dataframe(df),
    column_config={
        "Start Time": st.column_config.DatetimeColumn(
            None,
            min_value=datetime(2023, 6, 1),
            max_value=datetime(2025, 1, 1),
            # format="MMM D, YYYY, h:mm a",
            # format="YYYY-MM-DD, h:mm a",
            format="M/D/YY, h:mm a",
            step=60
        ),     
        "End Time": st.column_config.DatetimeColumn(
            None,
            min_value=datetime(2023, 6, 1),
            max_value=datetime(2025, 1, 1),
            # format="MMM D, YYYY, h:mm a",
            # format="YYYY-MM-DD, h:mm a",
            format="M/D/YY, h:mm a",
            step=60
        ),     
        "Event Page": st.column_config.LinkColumn(
            None, display_text="Event page"
        ),       
        "Categories": st.column_config.ListColumn(
            None,
            help="The sales volume in the last 6 months",
            width="large",
        ),
    }
)
# st.dataframe(df)

# Print results.

# for row in df.itertuples():
#     st.write(f"__{row.title}__:")
#     st.write(f"{row.description}")
