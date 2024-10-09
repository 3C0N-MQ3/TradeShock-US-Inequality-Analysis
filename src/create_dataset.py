"""
Creating the dataset:

Loading and merging data from 1990, 2000 and 2008 Census and ACS.

"""

# %%
import os
import sqlite3
import pandas as pd
import numpy as np

from toolz import pipe
from econtools import group_id

from src.write_variable_labels import write_variable_labels


def create_base_df(mainp):
    mainp = os.path.join("data")
    # This path will work relatively to the root of the project.
    mainp = os.path.join("data")
    print(f"--> Path to data: {mainp}")

    group_cols = ["male", "native", "agebin", "educbin", "white"]
    print("--> Group columns defined.")
    print(group_cols)

    write_variable_labels()

    # Keep those aged 16-64 and not in group quarters:
    df = pipe(
        os.path.join(mainp, "usa_00137.dta"),
        lambda x: pd.read_stata(x, convert_categoricals=False),
        lambda x: x[(x.age >= 16) & (x.age <= 64) & (x.gq <= 2)],
    )

    # Define (128) groups over which we CA:
    #    - gender (2)
    #    - US born (2)
    #    - age bin (4)
    #    - education bin (4)
    #    - race bin (2)

    df["male"] = np.where(df.sex == 1, 1, 0)
    df["native"] = np.where(df.bpl <= 99, 1, 0)
    df["agebin"] = pd.cut(df.age, bins=[15, 27, 39, 51, 64], labels=False)
    df["educbin"] = pd.cut(df.educ, bins=[-1, 5, 6, 9, 11], labels=False)
    df["white"] = np.where(df.race == 1, 1, 0)
    df["college"] = np.where((df.educ > 9) & (df.educ <= 11), 1, 0)

    df.drop(columns=["age", "educ", "race", "bpl", "sex"], inplace=True)
    print("--> Groups created.")

    df = group_id(df, cols=group_cols, merge=True, name="groups")
    print("--> Groups id created.")

    # Get geography to cz level
    # Katrina data issue
    df.loc[(df.statefip == 22) & (df.puma == 77777), "puma"] = 1801

    df["PUMA"] = df["statefip"].astype(str).str.zfill(2) + df["puma"].astype(
        str
    ).str.zfill(4)

    df["PUMA"] = df["PUMA"].astype("int")

    df = df.rename(columns={"puma": "puma_original"})

    df1990 = df[df.year == 1990].merge(
        pd.read_stata(os.path.join(mainp, "cw_puma1990_czone.dta")),
        left_on="PUMA",
        right_on="puma1990",
    )

    df2000 = df[df.year != 1990].merge(
        pd.read_stata(os.path.join(mainp, "cw_puma2000_czone.dta")),
        left_on="PUMA",
        right_on="puma2000",
    )

    df = pd.concat([df1990, df2000])
    df["perwt"] = df["perwt"] * df["afactor"]

    del df1990
    del df2000
    print("--> 1990 and 2000 census information merged.")

    # #### Aggregate to cz x group x year level
    # Employment status:
    df["emp"] = np.where(df.empstat == 1, 1, 0)
    df["unemp"] = np.where(df.empstat == 2, 1, 0)
    df["nilf"] = np.where(df.empstat == 3, 1, 0)
    print("--> Employment indicator columns created.")

    # Manufacturing employment:
    df["manuf"] = np.where(
        (df.emp == 1) & (df.ind1990 >= 100) & (df.ind1990 < 400), 1, 0
    )
    df["nonmanuf"] = np.where(
        (df.emp == 1) & ((df.ind1990 < 100) | (df.ind1990 >= 400)), 1, 0
    )
    print("--> Manufacturer and non-manufacturer indicator columns created.")

    # Filling in weeks worked for 2008 ACS (using midpoint):
    df.loc[df.wkswork2 == 1, "wkswork1"] = 7
    df.loc[df.wkswork2 == 2, "wkswork1"] = 20
    df.loc[df.wkswork2 == 3, "wkswork1"] = 33
    df.loc[df.wkswork2 == 4, "wkswork1"] = 43.5
    df.loc[df.wkswork2 == 5, "wkswork1"] = 48.5
    df.loc[df.wkswork2 == 6, "wkswork1"] = 51
    print("--> Weeks worked for 2008 ACS filled with midpoints.")

    # Log weekly wage:
    df["lnwkwage"] = np.log(df.incwage / df.wkswork1)
    df.loc[df["lnwkwage"] == -np.inf, "lnwkwage"] = np.nan
    print("--> Log of weekly wages created.")

    # Hours:
    df["hours"] = df["uhrswork"] * df["wkswork1"]
    print("--> Hours worked created.")

    df.drop(columns=["empstat", "wkswork2", "incwage"], inplace=True)

    return df, group_cols


def create_sql(mainp):

    df, _ = create_base_df()

    # Creating the sql database.
    conn = sqlite3.connect(os.path.join(mainp, "dataset.db"))

    try:
        df.to_sql("census", conn, if_exists="replace", index=False)
        print("DataFrame successfully written to the database.")
    except Exception as e:
        print(f"An error occurred: {e}")


# %%
if __name__ == "__main__":
    mainp = os.path.join("data")
    create_sql(mainp)
