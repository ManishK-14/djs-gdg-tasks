import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# --- Page settings ---
st.set_page_config(
    page_title="F1 Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title(" F1 Data Analysis & Modeling")
st.markdown("Upload the f1 CSV and explore data, plots, and predictive modeling in one place.")

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload F1 CSV", type=["csv"])

if uploaded_file is not None:
    f1_comp = pd.read_csv(uploaded_file, index_col=0)

    tab1, tab2, tab3 = st.tabs(["Data Preview", "Plots", "Modeling"])
    with tab1:
        st.subheader("Raw Data")
        st.dataframe(f1_comp.head())

        st.subheader("Summary Statistics")
        st.write(f1_comp.describe())

        st.subheader("Missing Values")
        f1_comp.replace('/N', np.nan, inplace=True)
        f1_comp['points'] = pd.to_numeric(f1_comp['points'], errors='coerce')
        f1_comp['laps'] = pd.to_numeric(f1_comp['laps'], errors='coerce')
        f1_comp['milliseconds'] = pd.to_numeric(f1_comp['milliseconds'], errors='coerce')
        f1_comp['fastestLapSpeed'] = pd.to_numeric(f1_comp['fastestLapSpeed'], errors='coerce')

        st.write(f1_comp.isnull().sum())

        # Handling missing values
        f1_comp['points'] = f1_comp['points'].fillna(f1_comp['points'].mean())
        f1_comp['laps'] = f1_comp['laps'].fillna(f1_comp['laps'].mode()[0])
        f1_comp['milliseconds'] = f1_comp['milliseconds'].fillna(f1_comp['milliseconds'].median())
        f1_comp['fastestLapSpeed'] = f1_comp['fastestLapSpeed'].fillna(f1_comp['fastestLapSpeed'].mean())

        st.write("After Filling Missing Values")
        st.write(f1_comp.isnull().sum())

    #Tab 2: Plots
    with tab2:
        st.subheader("Visualizations")
        plot_options = st.multiselect(
            "Select plots to display",
            options=[
                "Points Histogram",
                "Race Completion Boxplot",
                "Position vs Points",
                "Average Points per Year",
                "Top 10 Drivers",
                "Target Finish Distribution",
                "Fastest Lap Speed",
                "Top 10 Nationalities"
            ],
            default=["Points Histogram", "Top 10 Drivers"]
        )

        if "Points Histogram" in plot_options:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.hist(f1_comp['points'], bins=30, alpha=0.6, color='skyblue', edgecolor='black')
            sns.kdeplot(f1_comp['points'], color='red', linewidth=2, ax=ax)
            ax.set_xlabel("Points")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Points")
            st.pyplot(fig)

        if "Race Completion Boxplot" in plot_options:
            fig, ax = plt.subplots()
            sns.boxplot(x=f1_comp['milliseconds'], color='lightgreen', ax=ax)
            ax.set_title("Boxplot of Race Completion Time (ms)")
            st.pyplot(fig)

        if "Position vs Points" in plot_options:
            fig, ax = plt.subplots()
            ax.scatter(f1_comp['positionOrder'], f1_comp['points'], color='orange')
            ax.set_xlabel("Position Order")
            ax.set_ylabel("Points")
            ax.set_title("Position vs Points")
            st.pyplot(fig)

        if "Average Points per Year" in plot_options:
            points_by_year = f1_comp.groupby('year')['points'].mean()
            fig, ax = plt.subplots()
            ax.plot(points_by_year.index, points_by_year.values, marker='o', color='red')
            ax.set_xlabel("Year")
            ax.set_ylabel("Average Points")
            ax.set_title("Average Points per Year")
            st.pyplot(fig)

        if "Top 10 Drivers" in plot_options:
            top_drivers = f1_comp.groupby('surname')['points'].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=top_drivers.values, y=top_drivers.index, palette='cool', ax=ax)
            ax.set_xlabel("Total Points")
            ax.set_ylabel("Driver")
            ax.set_title("Top 10 Drivers by Total Points")
            st.pyplot(fig)

        if "Target Finish Distribution" in plot_options and 'target_finish' in f1_comp.columns:
            fig, ax = plt.subplots()
            sns.countplot(x='target_finish', data=f1_comp, palette='pastel', ax=ax)
            ax.set_title("Target Finish Distribution")
            st.pyplot(fig)

        if "Fastest Lap Speed" in plot_options:
            fig, ax = plt.subplots()
            sns.boxplot(x='fastestLapSpeed', data=f1_comp, color='yellow', ax=ax)
            ax.set_title("Distribution of Fastest Lap Speed")
            st.pyplot(fig)

        if "Top 10 Nationalities" in plot_options:
            top_countries = f1_comp['nationality_x'].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=top_countries.values, y=top_countries.index, palette='pastel', ax=ax)
            ax.set_xlabel("Number of Drivers")
            ax.set_title("Top 10 Driver Nationalities")
            st.pyplot(fig)

    # Task 3: Modeling 
    with tab3:
        if 'target_finish' in f1_comp.columns:
            st.subheader("Logistic Regression: Predict Target Finish")

            all_features = ['year', 'laps', 'points', 'fastestLapSpeed', 'lat', 'lng', 'alt', 'grid']
            selected_features = st.multiselect("Select features for modeling", all_features, default=all_features)

            if len(selected_features) > 0:
                X = f1_comp[selected_features]
                y = f1_comp['target_finish']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                scaler = StandardScaler()
                X_train[selected_features] = scaler.fit_transform(X_train[selected_features])
                X_test[selected_features] = scaler.transform(X_test[selected_features])

                model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_pred_prob = model.predict_proba(X_test)[:,1]

                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
                st.metric("ROC-AUC Score", f"{roc_auc_score(y_test, y_pred_prob):.3f}")

                st.write("### Confusion Matrix")
                st.write(confusion_matrix(y_test, y_pred))

                st.write("### Classification Report")
                st.text(classification_report(y_test, y_pred))

        else:
            st.info("No 'target_finish' column found for modeling.")



