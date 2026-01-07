# fuzzy.py
import numpy as np

EPS = 1e-9


def trapmf(x, a, b, c, d):
    """Robust trapezoidal membership function."""
    if a > b or c > d:
        return 0.0

    left = (x - a) / (b - a + EPS)
    right = (d - x) / (d - c + EPS)

    return float(max(0.0, min(1.0, min(left, right))))


class Fuzzy4:
    LOW, MED, HIGH = 20.0, 50.0, 85.0

    def _sanitize(self, v):
        return 0.0 if v is None or np.isnan(v) else float(v)

    def score(self, limit_bal, pay_0, bill1, pay_amt1):
        limit_bal = self._sanitize(limit_bal)
        pay_0     = self._sanitize(pay_0)
        bill1     = self._sanitize(bill1)
        pay_amt1  = self._sanitize(pay_amt1)

        # ---- LIMIT_BAL ----
        L_low  = trapmf(limit_bal, 0, 0, 50000, 150000)
        L_med  = trapmf(limit_bal, 50000, 150000, 300000, 600000)
        L_high = trapmf(limit_bal, 300000, 600000, 900000, 1000000)

        # ---- PAY_0 ----
        P_good = trapmf(pay_0, -3, -2, 0, 1)
        P_late = trapmf(pay_0, 0, 1, 2, 3)
        P_bad  = trapmf(pay_0, 2, 3, 8, 8)

        # ---- BILL_AMT1 ----
        B_low  = trapmf(bill1, -200000, 0, 20000, 80000)
        B_med  = trapmf(bill1, 20000, 80000, 200000, 400000)
        B_high = trapmf(bill1, 200000, 400000, 900000, 900000)

        # ---- PAY_AMT1 ----
        PA_none = trapmf(pay_amt1, 0, 0, 1000, 5000)
        PA_some = trapmf(pay_amt1, 1000, 5000, 20000, 50000)
        PA_full = trapmf(pay_amt1, 20000, 50000, 300000, 1000000)

        # ---- RULES ----
        lows = [
            min(L_high, P_good, PA_full),
            min(L_med, P_good, PA_some),
        ]

        meds = [
            min(L_med, P_late, B_med),
            min(L_low, P_good, B_med),
        ]

        highs = [
            min(L_low, P_bad, B_high),
            min(L_med, P_bad, B_high),
            min(L_low, P_late, PA_none),
        ]

        num = (
            sum(r * self.LOW  for r in lows) +
            sum(r * self.MED  for r in meds) +
            sum(r * self.HIGH for r in highs)
        )

        den = sum(lows) + sum(meds) + sum(highs)

        if den <= 0:
            return 45.0

        score = num / (den + EPS)
        return float(max(0.0, min(100.0, score)))
