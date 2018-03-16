import beta as b

# decimals are percentages
risk_free_rate = 2.39 / 100.00
market_return = 12.0 / 100.00
stock_beta = b.findBeta()


def getCAPM():
    expected_return = risk_free_rate + stock_beta * (market_return - risk_free_rate)
    return expected_return


print(getCAPM())
