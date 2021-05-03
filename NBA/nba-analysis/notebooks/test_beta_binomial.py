from nba_analysis.pipelines.data_science.nodes import BetaBinomial

dist = BetaBinomial()
dist.plot_pdf()

dist.bayes_update(trials=df_temp["FT_Attempted"].iloc[0], successes=df_temp["FT_Scored"].iloc[0])
dist.plot_pdf()

dist.bayes_update(trials=df_temp["FT_Attempted"].iloc[1], successes=df_temp["FT_Scored"].iloc[1])
dist.plot_pdf()


dist.alpha
dist.beta
df_temp["FT_Scored"].iloc[0]/df_temp["FT_Attempted"].iloc[0]