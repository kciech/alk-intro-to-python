"""Group 6 - A consultant's model has a tiny error. Should we buy it?"""
import pandas as pd
import streamlit as st

from ml_base import load_daily
from w03_group6_model_judge import ModelJudge

st.title("⚖️ A consultant's model has a tiny error. Should we buy it?")
st.caption("Card 6 · Team: Alex, Lukasz, Adam")

daily = load_daily()
judge = ModelJudge(daily, last_train_day="2011-09-30", max_degree=8).fit()
table = judge.results_
best = judge.best_degree()
baseline_mae = round((daily.loc["2011-10-01":, "revenue"] - daily.loc[:"2011-09-30", "revenue"].mean()).abs().mean())

st.header("1️⃣ What we built")
st.markdown(
    """
`ModelJudge(daily, last_train_day, max_degree)` checks a **polynomial trend** model: daily revenue as a curve of time
(`np.polyfit(years, revenue, degree)`, years = days since 1 Dec 2010 / 365).

- `.fit()` → fits every degree 1 … 8 on **Dec 2010 - Sep 2011** and measures the mean absolute error (MAE) on those days
  (**train**) and on **Oct - Dec 2011** (**test** = the months the model has never seen)
- `.best_degree()` → the degree with the lowest **test** error
- `.cv_mae(degree, method)` → 5-fold cross-validation, **random** folds or **time** folds
- `.plot(degree)` → the data, the curve and the last training day
"""
)

st.header("2️⃣ Our answer to the board")
col1, col2, col3 = st.columns(3)
col1.metric("Consultant's model (degree 8): train MAE", f"£{table.loc[8, 'train_mae']:,}")
col2.metric("The same model on the next months", f"£{table.loc[8, 'test_mae']:,}",
            delta=f"{table.loc[8, 'test_mae'] / table.loc[8, 'train_mae']:.1f}× worse", delta_color="inverse")
col3.metric(f"Best on the next months: degree {best}", f"£{table.loc[best, 'test_mae']:,}",
            delta=f"baseline (train mean): £{baseline_mae:,}", delta_color="off")
st.markdown(
    f"""
**Don't buy it on the £8,700 claim.** That is the error on the days the curve was *fitted on*. Higher degrees
always make the training error smaller, but on the next months degree 8 is off by **£{table.loc[8, 'test_mae']:,}**
a day, and degrees 4, 6 and 7 are much worse. A simple curve (degree {best}) does best on the new months, and even
that only beats "always predict the average" (£{baseline_mae:,}) by a little.
"""
)
st.subheader("All degrees")
st.dataframe(table.style.format("£{:,}").highlight_min(subset="test_mae", color="#c8e6c9"))
st.line_chart(table, x_label="degree", y_label="MAE (GBP)")

st.header("🎛️ Try it yourself")
degree = st.slider("Degree of the curve", 1, judge.max_degree, 8)
st.pyplot(judge.plot(degree).figure)
c1, c2 = st.columns(2)
c1.metric("Train MAE (Dec 2010 - Sep 2011)", f"£{table.loc[degree, 'train_mae']:,}")
c2.metric("Test MAE (Oct - Dec 2011)", f"£{table.loc[degree, 'test_mae']:,}")

st.subheader("Random vs time cross-validation")
st.markdown(
    "**Random** folds: the test days lie *between* training days, so the curve only has to fill gaps. "
    "**Time** folds: always train on the past, test on the *next* days - like real forecasting."
)
cv = pd.DataFrame({
    "random CV MAE": [judge.cv_mae(d, method="random") for d in range(1, judge.max_degree + 1)],
    "time CV MAE": [judge.cv_mae(d, method="time") for d in range(1, judge.max_degree + 1)],
}, index=pd.Index(range(1, judge.max_degree + 1), name="degree"))
st.dataframe(cv.style.format("£{:,}"))
st.markdown(
    f"For degree {degree}: random CV says **£{cv.loc[degree, 'random CV MAE']:,}**, "
    f"time CV says **£{cv.loc[degree, 'time CV MAE']:,}**."
)

st.header("3️⃣ Is our number real? + what the AI got wrong")
st.markdown(
    """
- **Only 60 test days**, and they are the **Christmas season** (Oct - Dec), which the training data has only once
  (December 2010). Any trend model is surprised by it - a different split date would give different numbers.
- A polynomial is just a curve: it has no idea about weekdays or seasons, and outside the training days it shoots
  up or down. The board should ask: **"What is the error on data the model has NOT seen - split by time?"**
- **AI:** The AI's code was wrong or misleading in one concrete moment, e.g. it said "degree 8 is best" when it was actually worse on the test months. Also, it forgot to run the setup code at the beginning, so the ModelJudge class was not defined.
"""
)
