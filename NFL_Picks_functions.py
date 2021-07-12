def moneyline_prob(OddsHome, OddsAway):

    #Returns the probability of the home team winning

    return (1/OddsHome)/(1/OddsHome + 1/OddsAway)
