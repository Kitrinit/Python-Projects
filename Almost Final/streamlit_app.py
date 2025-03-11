"""
STREAMLIT_APP.PY
================
Run this with:
    streamlit run streamlit_app.py

What it does:
  1) Loads the final trained model (no time_major).
  2) Loads label encoders for user1 and user2 states.
  3) Creates a synthetic day (e.g. Monday => 2880 steps of time).
  4) Predicts user1 & user2 states.
  5) Plots user1, user2, plus a 0/1 "together" line if they're in the same state.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ========== 1) LOAD MODEL & ENCODERS ==========
model = tf.keras.models.load_model("full_week_lstm_model.keras")

with open("le_state_user1.pkl","rb") as f:
    le_state_user1 = pickle.load(f)
with open("le_state_user2.pkl","rb") as f:
    le_state_user2 = pickle.load(f)

# ========== 2) STREAMLIT UI SETUP ==========
st.title("Full Week Activity Forecast")

st.write("Select a day below, then click 'Predict' to see the model's forecasts for both users, plus whether they're in the same location.")

day_options = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_num_map = {
    "Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
    "Friday":4,"Saturday":5,"Sunday":6
}
selected_day = st.selectbox("Choose a day to forecast:", day_options)

if st.button("Predict Activity Timeline"):
    # ========== 3) GENERATE SYNTHETIC INPUT ==========
    # For a 24-hour period @ 30s intervals => 2880 time steps
    time_steps = 2880
    t = np.arange(time_steps)
    day_num = day_num_map[selected_day]

    # Convert steps to hours & minutes
    hours = ((t*30)/3600) % 24
    minutes = ((t*30)/60) % 60

    # Cyclical encoding
    hour_sin = np.sin(2*np.pi*hours/24)
    hour_cos = np.cos(2*np.pi*hours/24)

    minute_sin = np.sin(2*np.pi*minutes/60)
    minute_cos = np.cos(2*np.pi*minutes/60)

    day_sin = np.sin(2*np.pi*day_num/7)*np.ones(time_steps)
    day_cos = np.cos(2*np.pi*day_num/7)*np.ones(time_steps)

    # is_weekend -> 1 if day_num in [5,6], else 0
    is_weekend_val = int(day_num in [5,6])
    weekend_array = np.full(time_steps, is_weekend_val)

    # Example order: [is_weekend, hour_sin, hour_cos, minute_sin, minute_cos, day_sin, day_cos]
    # Adjust to match exactly what your final training code produced
    X_day = np.stack([
        weekend_array,
        hour_sin, hour_cos,
        minute_sin, minute_cos,
        day_sin, day_cos
    ], axis=1)

    # Reshape to (batch=1, timesteps=2880, features=7)
    X_day = X_day[np.newaxis, :, :]

    # ========== 4) PREDICT ==========
    # Model has 2 outputs: (pred_user1_probs, pred_user2_probs)
    pred_user1_probs, pred_user2_probs = model.predict(X_day)

    # Convert to integer class IDs
    user1_int = np.argmax(pred_user1_probs, axis=-1).flatten()
    user2_int = np.argmax(pred_user2_probs, axis=-1).flatten()

    # Inverse transform to actual string states
    user1_states = le_state_user1.inverse_transform(user1_int)
    user2_states = le_state_user2.inverse_transform(user2_int)

    # ========== 5) DERIVE "TOGETHER" ==========
    # 1 if user1_state == user2_state, else 0
    is_together = np.array([
        int(u1 == u2) for (u1,u2) in zip(user1_states, user2_states)
    ])

    # ========== 6) PLOT RESULTS ==========
    st.subheader("User1 Predicted States")

    fig1, ax1 = plt.subplots(figsize=(10,3))
    unique_u1 = sorted(set(user1_states))
    user1_num = [unique_u1.index(s) for s in user1_states]
    ax1.step(range(len(user1_num)), user1_num, where="post", label="User1 State")
    ax1.set_xlabel("Time Step (30s intervals)")
    ax1.set_ylabel("State Index")
    ax1.set_yticks(range(len(unique_u1)))
    ax1.set_yticklabels(unique_u1)
    ax1.set_title(f"User1 Predicted States ({selected_day})")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("User2 Predicted States")

    fig2, ax2 = plt.subplots(figsize=(10,3))
    unique_u2 = sorted(set(user2_states))
    user2_num = [unique_u2.index(s) for s in user2_states]
    ax2.step(range(len(user2_num)), user2_num, where="post", label="User2 State")
    ax2.set_xlabel("Time Step (30s intervals)")
    ax2.set_ylabel("State Index")
    ax2.set_yticks(range(len(unique_u2)))
    ax2.set_yticklabels(unique_u2)
    ax2.set_title(f"User2 Predicted States ({selected_day})")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Are They Together?")

    fig3, ax3 = plt.subplots(figsize=(10,3))
    ax3.step(range(len(is_together)), is_together, where="post", label="Together=1, Not=0")
    ax3.set_yticks([0,1])
    ax3.set_yticklabels(["Not together","Together"])
    ax3.set_xlabel("Time Step (30s intervals)")
    ax3.set_title(f"Togetherness on {selected_day}")
    ax3.legend()
    st.pyplot(fig3)

    st.success("Prediction complete! Check the plots above.")
