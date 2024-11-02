# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
# ---

# %% [markdown]
"""
- Assignment 1: ADH (2013) with composition adjustment
- By: Augusto Ospital. 
- First version: April 5, 2022.
- This version: October 2024
    - Modified by `3CON-MQ3` GitHub organization.
"""
# %%
import warnings
import os
import pandas as pd
import numpy as np

from io import StringIO
from linearmodels.iv import IV2SLS, compare

from src.agg import WtSum, WtMean
from src.create_dataset import create_base_igt

# %%
mainp = os.path.join("data")

df_cgy, group_cols = create_base_igt(mainp)

# %% [markdown]
# #### Aggregate to cz x year level

# %% [markdown]
# We now have a database at the level of the commuting zone ($i$) by year ($t$) by group ($g$). For the regressions we need data at the level of commuting zone by year ($it$). We will construct composition-adjusted measures as
#
# $$L_{it}^{CA} = \sum_g \bar{\theta}_{ig} L_{igt}$$
#
# where the time-invariant weights $\bar{\theta}_{ig}$ are the average across periods of hours weights:
#
# $$
# \bar{\theta}_{ig} = \frac{1}{3} \left( \theta_{ig1990}+ \theta_{ig2000}+ \theta_{ig2008}\right)
# $$
# where
# $$
# \theta_{igt} = hours_{igt} \Big/ \left( \sum_g hours_{igt} \right).
# $$
#
#
# Note that $\sum_g \bar{\theta}_{ig}=1$.
# Note that $\mathcal{G}$.

# %%
# Create weights
df_w = df_cgy.reset_index()[["czone", "year", "groups", "hours"]].copy()

# Deal with missing obs as zeros (which they are):
df_w = (
    df_w.set_index(["czone", "year", "groups"])
    .unstack(level=[1, 2], fill_value=0.0)
    .stack(level=[1, 2])
)
df_w["weight_cgt"] = df_w["hours"] / df_w.groupby(["czone", "year"])["hours"].transform(
    "sum"
)
df_w["weight_cg"] = df_w.groupby(["czone", "groups"])["weight_cgt"].transform("mean")

df_cgy = pd.concat(
    [
        df_cgy,
        df_w[["weight_cgt", "weight_cg"]].rename(
            columns={"weight_cg": "weight", "weight_cgt": "weight_non_adjusted"}
        ),
    ],
    axis=1,
)

del df_w

# %%
# import sqlite3 as sql
# conn = sql.connect('data/china_syndrome.db')
# df_cgy.to_sql('data_by_cz_groups_years',conn, if_exists='replace', index=True)
# conn.close()


# %%
# Create the average log wages across various aggregations within a czone x year
def fun(m):
    return WtMean(
        df_cgy.reset_index(),
        cols=["lnwkwage"],
        weight_col="weight",
        by_cols=["czone", "year"],
        mask=m,
    )


col_mask = df_cgy.reset_index().college == 1
ncol_mask = df_cgy.reset_index().college == 0
male_mask = df_cgy.reset_index().male == 1
female_mask = df_cgy.reset_index().male == 0

df_cy = pd.concat(
    [
        fun(None),
        fun(col_mask).rename(columns={"lnwkwage": "lnwkwage_col"}),
        fun(ncol_mask).rename(columns={"lnwkwage": "lnwkwage_ncol"}),
        fun(male_mask).rename(columns={"lnwkwage": "lnwkwage_male"}),
        fun(female_mask).rename(columns={"lnwkwage": "lnwkwage_female"}),
        fun(col_mask & male_mask).rename(columns={"lnwkwage": "lnwkwage_col_male"}),
        fun(col_mask & female_mask).rename(columns={"lnwkwage": "lnwkwage_col_female"}),
        fun(ncol_mask & male_mask).rename(columns={"lnwkwage": "lnwkwage_ncol_male"}),
        fun(ncol_mask & female_mask).rename(
            columns={"lnwkwage": "lnwkwage_ncol_female"}
        ),
    ],
    axis=1,
)

# %%
# Create CA shares
share_cols = ["manuf_share", "nonmanuf_share", "unemp_share", "nilf_share"]


def fun(m):
    return WtMean(
        df_cgy.reset_index(),
        cols=share_cols,
        weight_col="weight",
        by_cols=["czone", "year"],
        mask=m,
    )


col_mask = df_cgy.reset_index().college == 1
ncol_mask = df_cgy.reset_index().college == 0

df_cy = pd.concat(
    [
        df_cy,
        fun(None),
        fun(col_mask).add_suffix("_col"),
        fun(ncol_mask).add_suffix("_ncol"),
    ],
    axis=1,
)

# %%
# Create CA log counts
# (We are taking a weighted average of logs. One could alternatively take the log of weighted averages)
count_cols = ["lnmanuf", "lnnonmanuf", "lnemp", "lnunemp", "lnnilf", "lnpop"]
df_cy = pd.concat(
    [
        df_cy,
        WtMean(
            df_cgy.reset_index(),
            cols=count_cols,
            weight_col="weight",
            by_cols=["czone", "year"],
        ),
    ],
    axis=1,
)

# %%
df_cy.head()

# %% [markdown]
# #### Create 10-year equivalent changes

# %%
cols = df_cy.columns.to_list()

# Reshape to wide format:
df_cy = df_cy.reset_index().pivot_table(index="czone", columns="year")

# Compute decadal differences:
for c in cols:
    df_cy["D{}".format(c), 1990] = df_cy[c, 2000] - df_cy[c, 1990]
    df_cy["D{}".format(c), 2000] = (df_cy[c, 2008] - df_cy[c, 2000]) * (10 / 7)
# Reshape back to long format:
df_cy = df_cy.stack().drop(columns=cols)

# %%
df_cy.head()

# %% [markdown]
# #### Name variables to be consistent with the ADH replication file and merge the explanatory variables

# %%
for c in share_cols:
    df_cy["D{}".format(c)] = df_cy["D{}".format(c)] * 100.0
    df_cy["D{}_col".format(c)] = df_cy["D{}_col".format(c)] * 100.0
    df_cy["D{}_ncol".format(c)] = df_cy["D{}_ncol".format(c)] * 100.0

# Multiply by 100 b/c reports log points:
cols_mask = df_cy.columns.str.contains("Dln")
for c in df_cy.columns[cols_mask]:
    df_cy[c] = df_cy[c] * 100.0

ADHnames = {
    # outcome for Table 3
    "Dmanuf_share": "d_sh_empl_mfg",
    # outcomes for Table 5
    # panel A
    "Dlnmanuf": "lnchg_no_empl_mfg",
    "Dlnnonmanuf": "lnchg_no_empl_nmfg",
    "Dlnunemp": "lnchg_no_unempl",
    "Dlnnilf": "lnchg_no_nilf",
    # panel B
    "Dmanuf_share": "d_sh_empl_mfg",
    "Dnonmanuf_share": "d_sh_empl_nmfg",
    "Dunemp_share": "d_sh_unempl",
    "Dnilf_share": "d_sh_nilf",
    # panel C
    "Dmanuf_share_col": "d_sh_empl_mfg_edu_c",
    "Dnonmanuf_share_col": "d_sh_empl_nmfg_edu_c",
    "Dunemp_share_col": "d_sh_unempl_edu_c",
    "Dnilf_share_col": "d_sh_nilf_edu_c",
    # panel D
    "Dmanuf_share_ncol": "d_sh_empl_mfg_edu_nc",
    "Dnonmanuf_share_ncol": "d_sh_empl_nmfg_edu_nc",
    "Dunemp_share_ncol": "d_sh_unempl_edu_nc",
    "Dnilf_share_ncol": "d_sh_nilf_edu_nc",
    # outcomes for Table 6
    "Dlnwkwage": "d_avg_lnwkwage",
    "Dlnwkwage_male": "d_avg_lnwkwage_m",
    "Dlnwkwage_female": "d_avg_lnwkwage_f",
    "Dlnwkwage_col": "d_avg_lnwkwage_c",
    "Dlnwkwage_ncol": "d_avg_lnwkwage_nc",
    "Dlnwkwage_col_male": "d_avg_lnwkwage_c_m",
    "Dlnwkwage_col_female": "d_avg_lnwkwage_c_f",
    "Dlnwkwage_ncol_male": "d_avg_lnwkwage_nc_m",
    "Dlnwkwage_ncol_female": "d_avg_lnwkwage_nc_f",
}

df_cy.rename(columns=ADHnames, inplace=True)

# %%
df_cy.head()

# %%
# Original non-CA data:
df_NCA = pd.read_stata("data/workfile_china.dta")

# CA data:
CA_cols = [v for k, v in ADHnames.items()]
other_cols = df_NCA.columns.difference(CA_cols)

# %%
df_CA = pd.merge(
    df_cy,
    df_NCA[other_cols],
    left_on=["czone", "year"],
    right_on=["czone", "yr"],
    how="inner",
)

del df_cy

# %% [markdown]
# ## Run Regressions!


# %%
def MyIVreg(formula, df):
    res = IV2SLS.from_formula(formula, df, weights=df["timepwt48"]).fit(
        cov_type="clustered", clusters=df["statefip"]
    )

    return res


# %%
# pd.options.display.latex.repr = True


def CompareDF(x, fit_stats=["Estimator", "R-squared", "No. Observations"], keep=[]):
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        y = pd.read_csv(
            StringIO(compare(x, stars=True, precision="std_errors").summary.as_csv()),
            skiprows=1,
            skipfooter=1,
            engine="python",
        )
    z = pd.DataFrame(
        data=y.iloc[:, 1:].values,
        index=y.iloc[:, 0].str.strip(),
        columns=pd.MultiIndex.from_arrays(
            arrays=[y.columns[1:], y.iloc[0][1:]], names=["Model", "Dep. Var."]
        ),
    )
    if not keep:
        return pd.concat([z.iloc[11:], z.loc[fit_stats]])
    else:
        return pd.concat(
            [
                *[z.iloc[z.index.get_loc(v) : z.index.get_loc(v) + 2] for v in keep],
                z.loc[fit_stats],
            ]
        )


# %%
def Table3(df):
    regions = list(filter(lambda x: x.startswith("reg"), df.columns))
    controls = [
        ["t2"],
        ["t2", "l_shind_manuf_cbp"],
        ["t2", "l_shind_manuf_cbp"] + regions,
        ["t2", "l_shind_manuf_cbp", "l_sh_popedu_c", "l_sh_popfborn", "l_sh_empl_f"]
        + regions,
        ["t2", "l_shind_manuf_cbp", "l_task_outsource", "l_sh_routine33"] + regions,
        [
            "t2",
            "l_shind_manuf_cbp",
            "l_sh_popedu_c",
            "l_sh_popfborn",
            "l_sh_empl_f",
            "l_task_outsource",
            "l_sh_routine33",
        ]
        + regions,
    ]

    baseform = "d_sh_empl_mfg ~ [d_tradeusch_pw ~ d_tradeotch_pw_lag] + 1"
    models = {
        "({})".format(i + 1): " + ".join([baseform, *controls[i]])
        for i in range(len(controls))
    }
    res = {i: MyIVreg(m, df) for i, m in models.items()}

    baseform_first = "d_tradeusch_pw ~ d_tradeotch_pw_lag + 1"
    models_first = {
        "({})".format(i + 1): " + ".join([baseform_first, *controls[i]])
        for i in range(len(controls))
    }
    res_first = {i: MyIVreg(m, df) for i, m in models_first.items()}

    return res, res_first


# %%
def Table5(df):
    regions = list(filter(lambda x: x.startswith("reg"), df.columns))
    controls = [
        "t2",
        "l_shind_manuf_cbp",
        "l_sh_popedu_c",
        "l_sh_popfborn",
        "l_sh_empl_f",
        "l_sh_routine33",
        "l_task_outsource",
    ] + regions
    lhs = {
        #         'A':['lnchg_no_empl_mfg','lnchg_no_empl_nmfg','lnchg_no_unempl','lnchg_no_nilf','lnchg_no_ssadiswkrs'],
        #         'B':['d_sh_empl_mfg','d_sh_empl_nmfg','d_sh_unempl','d_sh_nilf','d_sh_ssadiswkrs'],
        "A": [
            "lnchg_no_empl_mfg",
            "lnchg_no_empl_nmfg",
            "lnchg_no_unempl",
            "lnchg_no_nilf",
        ],
        "B": ["d_sh_empl_mfg", "d_sh_empl_nmfg", "d_sh_unempl", "d_sh_nilf"],
        "C": [
            "d_sh_empl_mfg_edu_c",
            "d_sh_empl_nmfg_edu_c",
            "d_sh_unempl_edu_c",
            "d_sh_nilf_edu_c",
        ],
        "D": [
            "d_sh_empl_mfg_edu_nc",
            "d_sh_empl_nmfg_edu_nc",
            "d_sh_unempl_edu_nc",
            "d_sh_nilf_edu_nc",
        ],
    }
    models_a = {
        "({})".format(i + 1): " + ".join(
            [
                "{} ~ 1 + [d_tradeusch_pw ~ d_tradeotch_pw_lag]".format(lhs["A"][i]),
                *controls,
            ]
        )
        for i in range(len(lhs["A"]))
    }
    models_b = {
        "({})".format(i + 1): " + ".join(
            [
                "{} ~ 1 + [d_tradeusch_pw ~ d_tradeotch_pw_lag]".format(lhs["B"][i]),
                *controls,
            ]
        )
        for i in range(len(lhs["B"]))
    }
    models_c = {
        "({})".format(i + 1): " + ".join(
            [
                "{} ~ 1 + [d_tradeusch_pw ~ d_tradeotch_pw_lag]".format(lhs["C"][i]),
                *controls,
            ]
        )
        for i in range(len(lhs["C"]))
    }
    models_d = {
        "({})".format(i + 1): " + ".join(
            [
                "{} ~ 1 + [d_tradeusch_pw ~ d_tradeotch_pw_lag]".format(lhs["D"][i]),
                *controls,
            ]
        )
        for i in range(len(lhs["D"]))
    }

    res_a = {i: MyIVreg(m, df) for i, m in models_a.items()}
    res_b = {i: MyIVreg(m, df) for i, m in models_b.items()}
    res_c = {i: MyIVreg(m, df) for i, m in models_c.items()}
    res_d = {i: MyIVreg(m, df) for i, m in models_d.items()}

    return res_a, res_b, res_c, res_d


# %%
def Table6(df):
    regions = list(filter(lambda x: x.startswith("reg"), df.columns))
    controls = [
        "t2",
        "l_shind_manuf_cbp",
        "l_sh_popedu_c",
        "l_sh_popfborn",
        "l_sh_empl_f",
        "l_sh_routine33",
        "l_task_outsource",
    ] + regions
    lhs = {
        "A": ["d_avg_lnwkwage", "d_avg_lnwkwage_m", "d_avg_lnwkwage_f"],
        "B": ["d_avg_lnwkwage_c", "d_avg_lnwkwage_c_m", "d_avg_lnwkwage_c_f"],
        "C": ["d_avg_lnwkwage_nc", "d_avg_lnwkwage_nc_m", "d_avg_lnwkwage_nc_f"],
    }
    models_a = {
        "({})".format(i + 1): " + ".join(
            [
                "{} ~ 1 + [d_tradeusch_pw ~ d_tradeotch_pw_lag]".format(lhs["A"][i]),
                *controls,
            ]
        )
        for i in range(len(lhs["A"]))
    }
    models_b = {
        "({})".format(i + 1): " + ".join(
            [
                "{} ~ 1 + [d_tradeusch_pw ~ d_tradeotch_pw_lag]".format(lhs["B"][i]),
                *controls,
            ]
        )
        for i in range(len(lhs["B"]))
    }
    models_c = {
        "({})".format(i + 1): " + ".join(
            [
                "{} ~ 1 + [d_tradeusch_pw ~ d_tradeotch_pw_lag]".format(lhs["C"][i]),
                *controls,
            ]
        )
        for i in range(len(lhs["C"]))
    }
    res_a = {i: MyIVreg(m, df) for i, m in models_a.items()}
    res_b = {i: MyIVreg(m, df) for i, m in models_b.items()}
    res_c = {i: MyIVreg(m, df) for i, m in models_c.items()}

    return res_a, res_b, res_c


# %% [markdown]
# ### Table 3: Change in Manuf/Pop, Pooled Regressions with Controls

# %% [markdown]
# #### I. 1990–2007 stacked first differences

# %%
keep = [
    "d_tradeusch_pw",
    "l_shind_manuf_cbp",
    "l_sh_popedu_c",
    "l_sh_popfborn",
    "l_sh_empl_f",
    "l_task_outsource",
    "l_sh_routine33",
]
pd.options.display.float_format = "{:.3f}".format
CompareDF(Table3(df_CA)[0], keep=keep)
# %%
# Elimina caracteres no numéricos y convierte a NaN cualquier valor que no se pueda convertir
tmp = tmp.replace(r"[^0-9.-]", "", regex=True)
tmp = tmp.apply(pd.to_numeric, errors="coerce")

# Redondea a 3 decimales
tmp = tmp.round(3)

# %%
tmp


# %% [markdown]
# **Interpretation**. In Column 1 we are estimating
# $$ 100 \times \Delta L^m_{it} = \alpha + \beta \Delta IPW_{uit} + \gamma_t + e_{it} $$
# where $L^m_{it}$ is (manufacturing employment)/(working-age population) and  $IPW_{uit}$ is the import exposure per worker measured in 1,000s of dollars (see Appendix Table 1 of ADH). Then an estimate $\widehat{\beta}=-0.7871$ means that an exogenous increase of $1,000 in exposure per worker leads to a predicted decrease of 0.79 percentage points in manufacturing employment per working-age population.

# %%
# 2SLS by Frisch-Waugh-Lovell - Column 3 of Table 3
import statsmodels.api as sm

# Residualize on controls:
regions = list(filter(lambda x: x.startswith("reg"), df_CA.columns))
controls = ["t2", "l_shind_manuf_cbp"] + regions
W = sm.add_constant(df_CA[controls])
r_x = sm.WLS(df_CA["d_tradeusch_pw"], W, weights=df_CA["timepwt48"]).fit().resid
r_y = sm.WLS(df_CA["d_sh_empl_mfg"], W, weights=df_CA["timepwt48"]).fit().resid
r_z = sm.WLS(df_CA["d_tradeotch_pw_lag"], W, weights=df_CA["timepwt48"]).fit().resid

# Predict X with Z:
x_hat = sm.WLS(r_x, r_z, weights=df_CA["timepwt48"]).fit().predict()

# Regress Y on predicted X:
sm.WLS(r_y, x_hat, weights=df_CA["timepwt48"]).fit().summary()

# %% [markdown]
# #### II. 2SLS first stage estimates

# %%
CompareDF(Table3(df_CA)[1], keep=["d_tradeotch_pw_lag"], fit_stats=["R-squared"])

# %% [markdown]
# ### Table 5: Change in Employment, Unemployment and Non-Employment

# %%
results5a, results5b, results5c, results5d = Table5(df_CA)

# %% [markdown]
# #### Panel A. 100 × log change in population counts

# %%
CompareDF(results5a, keep=["d_tradeusch_pw"], fit_stats=[])

# %% [markdown]
# #### Panel B. Change in population shares

# %%
CompareDF(results5b, keep=["d_tradeusch_pw"], fit_stats=[])

# %% [markdown]
# #### College education

# %%
CompareDF(results5c, keep=["d_tradeusch_pw"], fit_stats=[])

# %% [markdown]
# #### No college education

# %%
CompareDF(results5d, keep=["d_tradeusch_pw"], fit_stats=[])

# %% [markdown]
# ### Table 6: Wage Changes

# %%
results6a, results6b, results6c = Table6(df_CA)

# %% [markdown]
# #### Panel A. All education levels

# %%
CompareDF(results6a, keep=["d_tradeusch_pw"], fit_stats=["R-squared"])

# %% [markdown]
# #### Panel B. College education

# %%
CompareDF(results6b, keep=["d_tradeusch_pw"], fit_stats=["R-squared"])

# %% [markdown]
# #### Panel C. No college education

# %%
CompareDF(results6c, keep=["d_tradeusch_pw"], fit_stats=["R-squared"])
