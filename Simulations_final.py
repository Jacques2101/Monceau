# Importation des librairies
from io import BytesIO
import requests

import numpy as np
import plotly.express as px
import streamlit as st
from empyrical import (
    cagr,
    annual_volatility,
    sharpe_ratio,
    value_at_risk,
    max_drawdown
    )
import pandas as pd

# Configuration de la page streamlit
st.set_page_config(layout="wide")


# fonctions rebalancement
def rebal_portef(prix, weight, frais_sous_jacent=None, frais_uc=0.0, prelevement_sociaux_euro=0.0, freq=12, seuil=None):
    # Inititalisation des variables
    weight_new = pd.DataFrame(columns=prix.columns, index=prix.index)
    portfolio_value = pd.Series(index=prix.index)
    rebalance = pd.Series(index=prix.index)

    rebalance.iloc[:] = 0
    weight_new.iloc[0] = weight
    portfolio_value.iloc[0] = 100
    rdt = prix.pct_change().fillna(0)
    rdt['Fonds Euro - Monceau'] = rdt['Fonds Euro - Monceau']*(1-prelevement_sociaux_euro) # Intégration des prélèvements coiaux sur le contrats en €
    rdt.loc[:, rdt.columns != 'Fonds Euro - Monceau'] = rdt.loc[:, rdt.columns != 'Fonds Euro - Monceau'] - frais_uc/12 # Intégration des FG sur les UC

    # Calcul des pondérations
    for i, months in enumerate(prix.index):
        if i == 0:
            pass
        else:
            if seuil is None:
                portfolio_value.iloc[i] = (1+ (weight_new.iloc[i - 1] * rdt.iloc[i]).sum()) * portfolio_value.iloc[i - 1]
                portfolio_rdt = (portfolio_value.iloc[i] / portfolio_value.iloc[i - 1] - 1)
                if prix.index.month[i] % freq != 0:
                    weight_new.iloc[i] = ((1 + rdt.iloc[i]) / (1 + portfolio_rdt)) * weight_new.iloc[i - 1]
                else:
                    rebalance.iloc[i] = 1.0
                    weight_new.iloc[i] = weight
            else:
                portfolio_value.iloc[i] = (1+ (weight_new.iloc[i - 1] * rdt.iloc[i]).sum()) * portfolio_value.iloc[i - 1]
                portfolio_rdt = (portfolio_value.iloc[i] / portfolio_value.iloc[i - 1] - 1                )
                if (100* np.max(np.abs(weight_new.iloc[i - 1].rename("Poids")- pd.DataFrame.from_dict(weight, orient="index", columns=["Poids"]).T))< seuil):
                    weight_new.iloc[i] = ((1 + rdt.iloc[i]) / (1 + portfolio_rdt)) * weight_new.iloc[i - 1]
                else:
                    rebalance.iloc[i] = 1.0
                    weight_new.iloc[i] = weight
    weight_new.columns = ["Weight " + col for col in weight_new.columns]
    portfolio_value = portfolio_value.rename("Perf_strat")
    rebalance = rebalance.rename("rebalance")
    return pd.concat([portfolio_value, rebalance, weight_new], axis=1)


def strat_investisst(profil, taux_sans_risque, period_invest=12, period_max=5 * 12):
    # Calcul des pondérations
    poids = np.array(
        [i / period_invest for i in range(period_invest)]
        + [1] * (period_max - period_invest)
    )
    perf_strat = pd.Series(index=profil.index)
    perf = pd.Series(index=profil.index)
    volat = pd.Series(index=profil.index)
    sharpe = pd.Series(index=profil.index)
    perf_profil = profil.pct_change()
    perf_taux_sans_risque = taux_sans_risque.pct_change()

    for i, months in enumerate(profil.index):
        if i < profil.shape[0] - period_max:
            perf_strat.iloc[i : i + period_max] = (
                poids * perf_profil.iloc[i : i + period_max]
                + (1 - poids) * perf_taux_sans_risque.iloc[i : i + period_max]
            )
            perf.iloc[i + period_max] = 100 * cagr(
                perf_strat.iloc[i : i + period_max], period="monthly"
            )
            volat.iloc[i + period_max] = 100 * annual_volatility(
                perf_strat.iloc[i : i + period_max], period="monthly"
            )
            sharpe.iloc[i + period_max] = sharpe_ratio(
                perf_strat.iloc[i : i + period_max], period="monthly"
            )
        else:
            pass
    return (
        perf.dropna().rename("Strat complète"),
        volat.dropna().rename("Volatilité"),
        sharpe.dropna().rename("Sharpe"),
    )

def load_google(code):
    url = f"https://drive.google.com/uc?export=download&id={code}"
    file = requests.get(url)
    bytesio = BytesIO(file.content)
    return pd.read_parquet(bytesio)

# Importation des bases de données
@st.cache_data
def lire_data():
    # https://drive.google.com/file/d/1-A6TYjdMcw3yf83zyzxbKF5hlwvemF7W/view?usp=sharing
    prix = load_google('1-A6TYjdMcw3yf83zyzxbKF5hlwvemF7W')
    return prix.resample("ME").last().dropna()


# lecture de la base de données
prix = lire_data()
rebal = {"Mensuel": 1, "Trimestriel": 3, "Semestriel": 6, "Annuel": 12, "Jamais": 13}

# Création des différents onglets
tab1, tab2, tab3, tab4= st.tabs(
    [
        "Comparaisons des indices",
        "Comparaison méthode de rebalancement",
        "Comparaison de stratégies",
        "Investissement programmé",
    ]
)

# Comparaison des indices
with tab1:
    st.subheader("**Statistiques sur les indices**")

    fig = px.line(100 * prix / prix.iloc[0], title="Evolution des indices")
    fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=-0.5,
            x=0,
            yanchor="bottom",
        )
        )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    lag = 12 * (
        st.number_input("**Période glissante**", min_value=1, max_value=10, value=3)
    )
    cols = st.columns(2)
    # Perf N ans glissants
    fig = px.line(
        prix.pct_change()
        .rolling(lag)
        .apply(lambda x: 100 * cagr(x, period="monthly"))
        .dropna(),
        # text=prix.columns.str.slice(15),
        title=f"Performances {lag//12} ans annualisées",
    )
    fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=-0.5,
            x=0,
            yanchor="bottom",
        )
    )

    cols[0].plotly_chart(fig, use_container_width=True)
    # Volatilité N ans glissants
    fig = px.line(
        prix.pct_change()
        .rolling(lag)
        .apply(lambda x: 100 * annual_volatility(x, period="monthly"))
        .dropna(),
        title=f"Volatilités {lag//12} ans annualisées",
    )
    fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=-0.5,
            x=0,
            yanchor="bottom",
        )
    )
    cols[1].plotly_chart(fig, use_container_width=True)

    # Sharpe N ans glissants
    fig = px.line(
        prix.loc[:,
            [
                "MSCI Europe NR",
                "Barclays Euro Aggregate Index EUR",
                "MSCI World Net TR EUR",
                "Refinitiv Global Focus Hedged CB - EUR",
                "Bloomberg EM USD Aggregate Total Return Index Value Hedged EUR",
                "Bloomberg Global High Yield Total Return Index Value Hedged EUR",
            ],
        ]
        .pct_change()
        .rolling(lag)
        .apply(lambda x: sharpe_ratio(x, period="monthly"))
        .dropna(),
        title=f"Sharpe {lag//12} ans",
    )
    fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=-0.5,
            x=0,
            yanchor="bottom",
        )
    )

    cols[0].plotly_chart(fig, use_container_width=True)
    # Max DD N ans glissants
    fig = px.line(
        prix.loc[:,
            [
                "MSCI Europe NR",
                "Barclays Euro Aggregate Index EUR",
                "MSCI World Net TR EUR",
                "Refinitiv Global Focus Hedged CB - EUR",
                "Bloomberg EM USD Aggregate Total Return Index Value Hedged EUR",
                "Bloomberg Global High Yield Total Return Index Value Hedged EUR",
            ],
        ]
        .pct_change()
        .rolling(lag)
        .apply(lambda x: 100 * max_drawdown(x))
        .dropna(),
        title=f"Max DD {lag//12} ans",
    )
    fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=-0.5,
            x=0,
            yanchor="bottom",
        )
    )
    cols[1].plotly_chart(fig, use_container_width=True)

    stat = (
        prix.pct_change()
        .agg(
            [
                lambda x: 100 * cagr(x, period="monthly"),
                lambda x: 100 * annual_volatility(x, period="monthly"),
                lambda x: sharpe_ratio(x, risk_free=0.01258 / 12, period="monthly"),
                lambda x: 100 * max_drawdown(x),
            ]
        )
        .set_axis(["Performance annuelle", "Volatilité", "Sharpe (taux sans risque 1.2%)", "Max DD"])
        .T
    )

    st.write("**Statistiques globales depuis 2000**")
    cols[0].dataframe(stat
                .style
                .format(precision=2),
                use_container_width=True)

    fig = px.scatter(stat,
                     x='Volatilité', 
                     y="Performance annuelle",
                     hover_name=stat.index,
                    #  labels={'volatilite': 'Volatilité', 'rendement': 'Performance', 'text':"Classe d'actifs"},
                     text=stat.index.str.slice(stop=25), 
                     )
    fig.update_traces(textposition='top center', 
                      hovertemplate="<b>Classe d'actifs: %{hovertext}</b> <br> <br>Rendement: %{y:.2f} <br>Volatilité: %{x:.2f}") 
    cols[1].plotly_chart(fig, 
                         use_container_width=True)

    st.divider()
    st.write('**Statistiques Performances annualisées sur 8 ans glissants**')
    stat = (
        prix
        .rolling(8*12)
        .apply(
                lambda x: 100 * cagr(x.pct_change(), period="monthly"),
        )
        .describe()
        .T
        [['min', '25%', '50%', '75%', 'max']]
    )
    st.dataframe(stat
                 .style
                 .format(precision=2), 
                 use_container_width=True)

    # Corrélation
    st.divider()
    corr = (
        prix[
            [
                "Barclays Euro Aggregate Index EUR",
                "Bloomberg Global High Yield Total Return Index Value Hedged EUR",
                "Bloomberg EM USD Aggregate Total Return Index Value Hedged EUR",
                "Refinitiv Global Focus Hedged CB - EUR",
                "MSCI Europe NR",
                "MSCI World Net TR EUR",
            ]
        ]
        .pct_change()
        .corr()
    )
    fig = px.imshow(
        corr,
        title="Matrice de corrélation",
        color_continuous_scale=["blue", "yellow", "red"],
        x=corr.index,
        y=corr.columns,
        text_auto=".2f",
        width=700,
        height=700,
        aspect="auto",
    )
    fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
    st.plotly_chart(fig, use_container_width=True)


# code tab2 : comparaison méthode de rebalancement
with tab2:
    st.subheader("**Entrée des pondérations :**")
    cols = st.columns(9)
    portfolio = {}
    for j, asset in enumerate(prix.iloc[:, [2, 5, 1, 6, 7, 4, 0, 3]].columns):
        asset_weight = (cols[j].number_input(f"{asset[:15]} (%)", min_value=0.0, step=5.0, value=12.5, key=f"portef{j}")/ 100)
        portfolio[asset] = asset_weight

    total_weight = sum(portfolio.values())
    if np.abs(total_weight - 1) >= 0.01:
        st.warning(f"**Attention : La somme des poids de la stratégie n'est pas égale à 100%. Elle est de {round(100*total_weight,2)}%**")
        
    if st.button('Calcul des stratégies', key='lancement_calcul_rebal'):
        # Calcul des stat sur différentes stratégie de rabalancement
        data1 = rebal_portef(prix, weight=portfolio, freq=1)  # rebal mensuel
        data2 = rebal_portef(prix, weight=portfolio, freq=3)  # rebal trimestriel
        data3 = rebal_portef(prix, weight=portfolio, freq=6)  # rebal semestriel
        data4 = rebal_portef(prix, weight=portfolio, freq=12)  # rebal annuel
        data5 = rebal_portef(prix, weight=portfolio, freq=13)  # pas de rebal
        data6 = rebal_portef(prix, weight=portfolio, seuil=1.0)  # rebal si seuil>1%
        data7 = rebal_portef(prix, weight=portfolio, seuil=2.0)  # rebal si seuil>2.0%
        data8 = rebal_portef(prix, weight=portfolio, seuil=3.0)  # rebal si seuil>3.0%

        lag = 12 * (st.number_input("**Période glissante**", min_value=1, max_value=10, value=3, key="lag"))

        fig = px.line(
            pd.concat(
                [
                    data1["Perf_strat"].rename("Reb. mensuel"),
                    data2["Perf_strat"].rename("Reb. trimestriel"),
                    data3["Perf_strat"].rename("Reb. semestriel"),
                    data4["Perf_strat"].rename("Reb. annuel"),
                    data5["Perf_strat"].rename("Pas de reb."),
                    data6["Perf_strat"].rename("Seuil 1%"),
                    data7["Perf_strat"].rename("Seuil 2%"),
                    data8["Perf_strat"].rename("Seuil 3%"),
                ],
                axis=1,
            ),
            title=f"Performance cumulée selon méthode de rebalancement",
        )
        fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
        fig.update_layout(
            legend=dict(
                orientation="h",
                y=-0.5,
                x=0,
                yanchor="bottom",)
            )
        

        cols = st.columns(2)
        cols[0].plotly_chart(fig, use_container_width=True)
        
        fig = px.line(
            pd.concat(
                [
                    data1["Perf_strat"].rename("Reb. mensuel"),
                    data2["Perf_strat"].rename("Reb. trimestriel"),
                    data3["Perf_strat"].rename("Reb. semestriel"),
                    data4["Perf_strat"].rename("Reb. annuel"),
                    data5["Perf_strat"].rename("Pas de reb."),
                    data6["Perf_strat"].rename("Seuil 1%"),
                    data7["Perf_strat"].rename("Seuil 2%"),
                    data8["Perf_strat"].rename("Seuil 3%"),
                ],
                axis=1,
            )
            .rolling(lag)
            .apply(lambda x: 100*cagr(x.pct_change(), period="monthly"))
            .dropna(),
            title=f"Performance anuelle moyenne {int(lag/12)} ans selon méthode de rebalancement",
        )
        fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
        fig.update_layout(
            legend=dict(
                orientation="h",
                y=-0.5,
                x=0,
                yanchor="bottom",
            )
        )

        cols[1].plotly_chart(fig, use_container_width=True)

        st.write("**Nombre de rabalancement par an :**")
        rebalct = (
            pd.concat(
                [
                    data1["rebalance"].rename("Reb. mensuel"),
                    data2["rebalance"].rename("Reb. trimestriel"),
                    data3["rebalance"].rename("Reb. semestriel"),
                    data4["rebalance"].rename("Reb. annuel"),
                    data5["rebalance"].rename("Pas de reb."),
                    data6["rebalance"].rename("Seuil 1%"),
                    data7["rebalance"].rename("Seuil 2%"),
                    data8["rebalance"].rename("Seuil 3%"),
                ],
                axis=1,
            )
            .resample("YE")
            .sum()
            .loc["2001":]
        )
        rebalct.index = rebalct.index.strftime("%Y")
        st.dataframe(rebalct, use_container_width=True)

        perf_rebal = (
            pd.concat(
                [
                    data1["Perf_strat"].rename("Reb. mensuel"),
                    data2["Perf_strat"].rename("Reb. trimestriel"),
                    data3["Perf_strat"].rename("Reb. semestriel"),
                    data4["Perf_strat"].rename("Reb. annuel"),
                    data5["Perf_strat"].rename("Pas de reb."),
                    data6["Perf_strat"].rename("Seuil 1%"),
                    data7["Perf_strat"].rename("Seuil 2%"),
                    data8["Perf_strat"].rename("Seuil 3%"),
                ],
                axis=1,
            )
            .agg(
                [
                    lambda x: 100 * cagr(x.pct_change(), period="monthly"),
                    lambda x: 100 * annual_volatility(x.pct_change(), period="monthly"),
                    lambda x: 100 * max_drawdown(x.pct_change()),
                    lambda x: 100 * value_at_risk(x.pct_change().dropna(), cutoff=0.05),
                    lambda x: sharpe_ratio(x.pct_change().dropna(), period='monthly', risk_free=1.26/12/100)
                ]
            )
            .set_axis(["Perf annuel moyenne", "Volat", "Max DD", "VaR 95%", 'Sharpe (taux sans risque 1.2%)'])
            .T
        )

        st.write("**Performance selon rebalancement**")
        st.dataframe(
            perf_rebal
            .style
            .format(precision=2)
            .highlight_max(color="red")
            .highlight_min(color="yellow")
            .highlight_max(subset=["Perf annuel moyenne", 'Sharpe (taux sans risque 1.2%)'], color="yellow")
            .highlight_min(subset=["Perf annuel moyenne", 'Sharpe (taux sans risque 1.2%)'], color="red"),
            use_container_width=True,
        )

# code tab 3 : comparaison de stratégie
with tab3:
    st.subheader("Stratégies :")
    cols = st.columns(6)
    num_portfolios = cols[0].number_input("**Nombre de portefeuilles**", min_value=1, step=1, value=1)
    choix = cols[1].selectbox("**Méthode pour rebalancer le portefeuille:**",
                              ["Mensuel", "Trimestriel", "Semestriel", "Annuel", "Jamais"],
                              key="rebal",
                              )
    periode = cols[2].number_input("**Période d'investissement (en année)**", 
                                   value=8,
                                   min_value=1,
                                   max_value=10,
                                   step=1)
    prelevement_sociaux_euro = cols[3].number_input("**Prélèvements sociaux contrat en € (en %)**", 
                                                    min_value=0.0, 
                                                    max_value=100.0, 
                                                    value=17.2)/100
    frais_uc = cols[4].number_input("**Frais sur les UC (en %)**", 
                                    min_value=0.0, 
                                    max_value=20.0,
                                    value=0.75)/100
    seuil = cols[5].number_input("**Seuil (en %)**", 
                                 value=5, 
                                 min_value=0, 
                                 max_value=100, 
                                 step=5, 
                                 key="seuil")/100

    methode = st.radio("**Méthode d'entrée des portefeuilles**", 
                       ["Pondérations de chaque classe d'actifs", 'Profil'], 
                       index=0,
                       horizontal=True)
    

    if methode=="Pondérations de chaque classe d'actifs":
        st.write("**Entrée l'allocation de chaque stratégie:**")
        portfolios = []
        cols = st.columns(8)
        for i in range(num_portfolios):
            portfolio = {}
            for j, asset in enumerate(prix.iloc[:, [2, 5, 1, 6, 7, 4, 0, 3]].columns):
                asset_weight = (
                    cols[j].number_input(
                        f"{asset[:15]} (%)",
                        min_value=0.,
                        step=5.,
                        value=12.5,
                        key=f"portef{i}{j}",)/100
                    )
                portfolio[asset] = asset_weight

            total_weight = sum(portfolio.values())
            if np.abs(total_weight - 1) >= 0.01:            
                st.warning(f"**Attention : La somme des poids de la stratégie {i+1} n'est pas égale à 100%. Elle est de {100*round(total_weight,2)}%**")
            portfolios.append(portfolio)
    else:
        portfolios = []
        st.write("**Entrée le profil type:**")
        cols = st.columns(7)
        asset_weight = {}
        for j, asset in enumerate(prix.iloc[:, [2, 1, 6, 7, 4, 0, 3]].columns):
            asset_weight[asset] = (cols[j].number_input(f"{asset[:20]} (%)",
                                                        min_value=0.,
                                                        step=5.,
                                                        value=100/7,
                                                        key=f"profil{j}",)/100
                            )
        if np.abs(sum(asset_weight.values())-1) >= 0.01: 
            st.warning(f"**La somme des poids du profil doit être égale à 100%. Elle est de {100*round(sum(asset_weight.values()),2)}%**")

        st.divider()
        st.write("**Poids du contrat en Euros dans chaque stratégie**")
        cols = st.columns(num_portfolios)
        for i in range(num_portfolios):
            portfolio = {}
            cols[i].write(f"**Stratégie {i+1}**")
            asset_weight_euro = (cols[i].number_input("Poids du contrat en € (%)",
                                                      min_value=0.,
                                                      step=5.,
                                                      value=50.0,
                                                      key=f"portef{i}{j}",)/100
                            )
            portfolio['Fonds Euro - Monceau'] = asset_weight_euro
            for asset in prix.iloc[:, [2, 1, 6, 7, 4, 0, 3]].columns: 
                portfolio[asset] = asset_weight[asset]*(1-asset_weight_euro)

            portfolios.append(portfolio)

        st.write(f"**Allocation des {num_portfolios} stratégies**")
        st.dataframe(pd.DataFrame(portfolios, 
                                  index=[f'Strategie {i+1}' for i in np.arange(len(portfolios))])
                     .replace(0,np.nan)
                     .dropna(axis=1,how="all")
                     .style
                     .format('{:.2%}'),
                     use_container_width=True, 
                     ) 


    if st.button('Calcul des stratégies', ):
        perf = [rebal_portef(prix, weight=weigth, freq=rebal[choix], prelevement_sociaux_euro=prelevement_sociaux_euro, frais_uc=frais_uc)["Perf_strat"] for weigth in portfolios]
        perf = pd.DataFrame(perf, index=["stratégie " + str(i+1) for i in range(len(perf))]).T 
        cols = st.columns(2)
        # Calcul de la perf cumulée
        fig = px.line(perf, title="Performanes cumulées")
        fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
        fig.update_layout(
            legend=dict(
                orientation="h",
                y=-0.5,
                x=0,
                yanchor="bottom",
            )
        )
        cols[0].plotly_chart(fig, use_container_width=True)

        # Calcul de la perf annuelle moyenne
        fig = px.line(
            perf.rolling(12 * periode)
            .apply(lambda x: 100 * cagr(x.pct_change(), period="monthly"))
            .dropna(),
            title=f"Performance annuelle sur {periode} années",
        )
        fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
        fig.update_layout(
            legend=dict(
                orientation="h",
                y=-0.5,
                x=0,
                yanchor="bottom",
            )
        )
        cols[1].plotly_chart(fig, use_container_width=True)

        # Calcul de volatilié
        fig = px.line(
            perf.rolling(12 * 5)
            .apply(lambda x: 100 * annual_volatility(x.pct_change(), period="monthly"))
            .dropna(),
            title=f"Volatilité annuelle 5 ans",
        )
        fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
        fig.update_layout(
            legend=dict(
                orientation="h",
                y=-0.5,
                x=0,
                yanchor="bottom",
            )
        )
        fig.add_hline(
            y=2,
            line_dash="dot",
            annotation_text="SRRI 3",
            annotation_font_color="red",
            line_color="red",
            line_width=3,
            annotation_position="top right",
        )
        fig.add_hline(
            y=5,
            line_dash="dot",
            annotation_text="SRRI 4",
            line_color="red",
            line_width=3,
            annotation_font_color="red",
            annotation_position="top right",
        )
        fig.add_hline(
            y=10,
            line_dash="dot",
            annotation_text="SRRI 5",
            line_color="red",
            line_width=3,
            annotation_font_color="red",
            annotation_position="top right",
        )
        fig.add_hline(
            y=15,
            line_dash="dot",
            annotation_text="SRRI 6",
            line_color="red",
            line_width=3,
            annotation_font_color="red",
            annotation_position="top right",
        )
        cols[0].plotly_chart(fig, use_container_width=True)

        # Calcul Sharpe
        fig = px.line(
            perf.rolling(12 * periode)
            .apply(lambda x: sharpe_ratio(x.pct_change(), period="monthly"))
            .dropna(),
            title=f"Ratio de Sharpe {periode} années",
        )
        fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
        fig.update_layout(
            legend=dict(
                orientation="h",
                y=-0.5,
                x=0,
                yanchor="bottom",
            )
        )
        cols[1].plotly_chart(fig, use_container_width=True)

        date = ['1 an', '2 ans', '3 ans', '4 ans', '5 ans', '6 ans', '7 ans', '8 ans', '9 ans', '10 ans', '11 ans', '12 ans']
        cone_perf_inf = [perf.rolling(lag).apply(lambda x: 100*cagr(x.pct_change(), period='monthly')).dropna().apply(lambda x: np.percentile(x, seuil*100))
                      for lag in np.arange(12, 12*(12+1), step=12)]
        cone_perf_inf = pd.DataFrame(cone_perf_inf)
        cone_perf_inf.index = date
        fig = px.line(cone_perf_inf, 
                      title=f"VaR à {100*seuil}% selon les stratégies")
        fig.update_traces(mode="markers+lines", hovertemplate="%{y:.2f}%")
        fig.update_layout(hovermode="x unified")
        fig.add_hline(
            y=0,
            line_dash="dot",   
            line_color="green",
            line_width=3
            )
        cols[0].plotly_chart(fig, use_container_width=True)

        cone_perf_sup = [perf.rolling(lag).apply(lambda x: 100*cagr(x.pct_change(), period='monthly')).dropna().apply(lambda x: np.percentile(x, 100-seuil*100))
                      for lag in np.arange(12, 12*(12+1), step=12)]
        cone_perf_sup = pd.DataFrame(cone_perf_sup)
        cone_perf_sup.index = date
        fig = px.line(cone_perf_sup, 
                      title=f"VaR à {100*(1-seuil)}% selon les stratégies")
        fig.update_traces(mode="markers+lines", hovertemplate="%{y:.2f}%")
        fig.update_layout(hovermode="x unified")
        fig.add_hline(
            y=0,
            line_dash="dot",   
            line_color="green",
            line_width=3
            )
        cols[1].plotly_chart(fig, use_container_width=True)
        

        # Tableau de synthèse
        st.write("**Statistiques des stratégies**")
        st.dataframe(
            perf.pct_change()
            .agg(
                [
                    lambda x: 100 * cagr(x, period="monthly"),
                    lambda x: 100 * annual_volatility(x, period="monthly"),
                    lambda x: sharpe_ratio(x, period="monthly", risk_free=1.26/12/100),
                    lambda x: 100 * max_drawdown(x),
                    lambda x: 100*value_at_risk(x.dropna(), cutoff=0.05)
                ])
            .set_axis(["Perf annuelle moyenne", "Volatilité moyenne", "Sharpe (taux sans risque 1.2%)", "Max DD", "VaR 95% 1 mois"])
            .T
            .style
            .format(precision=2),
            use_container_width=True,
            )

with tab4:
    st.subheader("**Entrée des pondérations :**")
    cols = st.columns(2)
    choix = cols[0].selectbox(
        "**Méthode pour rebalancer le portefeuille:**",
        ["Mensuel", "Trimestriel", "Semestriel", "Annuel", "Jamais"],
        key="rebal_final",
    )
    periode = cols[1].number_input(
        "**Période glissante (en année)**",
        value=8,
        min_value=1,
        max_value=8,
        step=1,
        key="periode_final",
    )

    portfolio = {}
    cols = st.columns(8)
    for j, asset in enumerate(prix.iloc[:, [2, 5, 1, 6, 7, 4, 0, 3]].columns):
        asset_weight = (
            cols[j].number_input(
                f"{asset[:25]} (%)",
                min_value=0.,
                step=5.,
                value=12.5,
                key=f"invt_prog_poids{j}",
            )
            / 100
        )
        portfolio[asset] = asset_weight

    total_weight = sum(portfolio.values())
    if np.abs(total_weight - 1) >= 0.01:
        st.warning(f"**Attention : La somme des poids de la stratégie n'est pas égale à 100%. Elle est de {round(100*total_weight,2)}%**")


    data = rebal_portef(prix, weight=portfolio, freq=rebal[choix])

    # Perf comparée des différentes stratégies
    perf = pd.concat(
        [
            strat_investisst(
                data["Perf_strat"],
                prix["Fonds Euro - Monceau"],
                period_invest=12,
                period_max=12 * periode,)[0].rename("Programme 1 an"),
            strat_investisst(
                data["Perf_strat"],
                prix["Fonds Euro - Monceau"],
                period_invest=24,
                period_max=12 * periode,)[0].rename("Programme 2 ans"),
            data["Perf_strat"]
            .rolling(periode * 12)
            .apply(lambda x: 100 * cagr(x.pct_change(), period="monthly"))
            .rename("Profil"),
        ],
        axis=1,).dropna()

    # Volat comparée des différentes stratégies
    volat = pd.concat(
        [
            strat_investisst(data['Perf_strat'],
                             prix['Fonds Euro - Monceau'],
                             period_invest=12, 
                             period_max=12*periode)[1].rename('Programme 1 an'),
            strat_investisst(data['Perf_strat'],
                             prix['Fonds Euro - Monceau'],
                             period_invest=24,
                             period_max=12*periode)[1].rename('Programme 2 ans'),
            data['Perf_strat']
            .rolling(periode*12)
            .apply(lambda x: 100*annual_volatility(x.pct_change(), period='monthly'))
            .rename('Profil')
            ],
        axis=1).dropna()

    cols = st.columns(2)

    fig = px.line(perf, title=f"Performance annuelle moyenne {periode} ans")
    fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=-0.5,
            x=0,
            yanchor="bottom",
        )
    )

    cols[0].plotly_chart(fig, use_container_width=True)
    fig = px.line(volat,  title=f"Volatilité annuelle moyenne {periode} ans")
    fig.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=-0.5,
            x=0,
            yanchor="bottom",
        )
    )
    cols[1].plotly_chart(fig, use_container_width=True)
