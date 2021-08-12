import numpy as np
import pandas as pd
from gekko import GEKKO


def submit(res, prc=1):
    discharge = pd.DataFrame(res).applymap(lambda x: np.round(x, prc) if x > 0 else 0).rename(columns={0: 'Discharge'})
    charge = pd.DataFrame(res).applymap(lambda x: np.round(-x, prc) if x < 0 else 0).rename(columns={0: 'Charge'})
    df = charge.merge(discharge, how='left', left_index=True, right_index=True)
    return df


def main():

    m = GEKKO()

    max_rate = 100.  # MW
    max_storage = 200.  # MWh
    nhr = 24

    data = pd.read_csv('../prediction/lgb_submission.csv', parse_dates=['Date'])
    price = data['Price ($/MWh)']

    # Battery Power
    bp = m.Array(m.Var, nhr)
    for i in range(nhr):
        if i == 0:
            bp[i].value = 0
        else:
            bp[i].value = 12
        bp[i].lower = -max_rate
        bp[i].upper = max_rate

    # 1 cycle fill and empty
    m.Equation(sum(bp) == 0)

    bp_abs = m.Array(m.Param, (nhr))
    for i in range(nhr):
        bp_abs[i] = m.abs2(bp[i])
    m.Equation(sum(bp_abs) == 2 * max_storage)

    # There should be enough power to sell at any time
    for i in range(1, nhr):
        m.Equation(bp[i] + sum(bp[:i - 1]) <= 0)

    m.Maximize(sum(price * bp))
    m.solve()

    result = [i.value[0] for i in bp]
    result2 = submit(result)
    result2.to_csv('schedule.csv', index=False)

    print('Expected Revenue: $%i' % sum(result * price))

    return result2


if __name__ == '__main__':
    main()
