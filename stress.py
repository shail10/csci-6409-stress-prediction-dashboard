import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import os
import pandas as pd
import dice_ml
import numpy as np

scaler = joblib.load("saved_models/scaler.joblib")
encoded_columns = joblib.load("saved_models/encoded_columns.joblib")
modelRandomForest = joblib.load(os.path.join("saved_models", "RandomForest.joblib"))


class_labels = ["No", "Maybe", "Yes"]


def plot_prediction_probabilities(probabilities, labels):
  # Define custom colors for the classes
  color_map = {
      "No": "#4CAF50",       
      "Maybe": "#FFC107",    
      "Yes": "#F44336"       
  }

  colors = [color_map[label] for label in labels]

  fig = go.Figure(data=[
      go.Pie(
          labels=labels,
          values=probabilities,
          hole=0.4,
          marker=dict(colors=colors)
      )
  ])

  fig.update_layout(
      title="Prediction Probabilities",
      annotations=[dict(text=f'{max(probabilities)*100:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
      showlegend=True,
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)',
      margin=dict(t=50, b=50, l=50, r=50)
  )

  return fig


def preprocess_input(input_df: pd.DataFrame):
  df_encoded = input_df.copy()

  low_cardinality_cols = [
      "Gender", "Occupation", "family_history", "treatment", "Days_Indoors",
      "Changes_Habits", "Mental_Health_History", "Mood_Swings", "Coping_Struggles", "Work_Interest",
      "Social_Weakness", "mental_health_interview"
  ]

  df_encoded = pd.get_dummies(df_encoded, columns=low_cardinality_cols)

  bool_cols = df_encoded.select_dtypes(include=['bool']).columns
  df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

  missing_cols = set(encoded_columns) - set(df_encoded.columns)
  for col in missing_cols:
      df_encoded[col] = 0  
      
  df_encoded = df_encoded[encoded_columns]

  # scaled = scaler.transform(df_encoded)
  # processed_input = pd.DataFrame(scaled, columns=encoded_columns)

  return df_encoded

# --- User input from sidebar ---
def user_input_features():
    st.sidebar.header("User Input - Stress Factors")
    data = {
        'Gender': st.sidebar.selectbox("Gender", ['Female', 'Male']),
        'Occupation': st.sidebar.selectbox("Occupation", ['Corporate', 'Student', 'Business', 'Housewife', 'Others']),
        'family_history': st.sidebar.selectbox("Family History of Mental Illness", ['No', 'Yes']),
        'treatment': st.sidebar.selectbox("Are you receiving treatment?", ['Yes', 'No']),
        'Days_Indoors': st.sidebar.selectbox("Days Spent Indoors", ['1-14 days', 'Go out Every day', 'More than 2 months', '15-30 days', '31-60 days']),
        'Changes_Habits': st.sidebar.selectbox("Changes in Habits", ['No', 'Yes', 'Maybe']),
        'Mental_Health_History': st.sidebar.selectbox("Past Mental Health History", ['Yes', 'No', 'Maybe']),
        'Mood_Swings': st.sidebar.selectbox("Mood Swings", ['Medium', 'Low', 'High']),
        'Coping_Struggles': st.sidebar.selectbox("Struggling to Cope", ['No', 'Yes']),
        'Work_Interest': st.sidebar.selectbox("Loss of Interest in Work", ['No', 'Maybe', 'Yes']),
        'Social_Weakness': st.sidebar.selectbox("Feeling Socially Weak", ['Yes', 'No', 'Maybe']),
        'mental_health_interview': st.sidebar.selectbox("Willing to Discuss Mental Health at Work", ['No', 'Maybe', 'Yes']),
    }

    return pd.DataFrame([data])


def generate_counterfactual_for_instance(model, input_instance, dataset_path, target_col='Growing_Stress', cfs_per_class=1):

  if not os.path.exists(dataset_path):
      raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

  dataset = pd.read_csv(dataset_path)

  dataset_for_dice = dataset

  categorical_features = list(input_instance.columns)

  dice_data = dice_ml.Data(
      dataframe=dataset_for_dice,
      continuous_features=[], 
      categorical_features=categorical_features,
      outcome_name=target_col
  )

  dice_model = dice_ml.Model(model=model, backend="sklearn")
  explainer = dice_ml.Dice(dice_data, dice_model, method="random")

  # Get current prediction
  original_pred = model.predict(input_instance.loc[0].to_frame().T,)[0]
  possible_classes = [0, 1, 2]
  desired_classes = [cls for cls in possible_classes if cls != original_pred]

  # Generate counterfactuals
  all_counterfactuals = []
  for desired_class in desired_classes:
    cf = explainer.generate_counterfactuals(
        input_instance.loc[0].to_frame().T,
        total_CFs=cfs_per_class,
        desired_class=desired_class,
        verbose=False
    )
    cf_df = cf.cf_examples_list[0].final_cfs_df
    all_counterfactuals.append(cf_df)

  # Combine all counterfactuals
  all_cf_df = pd.concat(all_counterfactuals, ignore_index=True)

  return all_cf_df


def multiclass_counterfactual_analysis_all_models(model, dataset_path, X_test, num_samples=50, cfs_per_instance=2, target_col='Growing_Stress'):
    """
    Generates model-level counterfactual analysis for all models in models_dict.

    Args:
        num_samples: Number of test samples to use.
        cfs_per_instance: Counterfactuals generated per desired class.
        target_col: Target column name.

    Returns:
        Dictionary of DataFrames containing counterfactual feature summaries per model.
    """
    categorical_features = [col for col in X_test.columns]

    results_summary = {}

    dice_data = dice_ml.Data(
        dataframe=pd.read_csv(dataset_path),
        continuous_features=[],
        categorical_features=categorical_features,
        outcome_name=target_col
    )

    dice_model = dice_ml.Model(model=model, backend="sklearn")
    explainer = dice_ml.Dice(dice_data, dice_model, method="random")

    feature_change_counts = {feature: 0 for feature in X_test.columns}
    samples_used = min(num_samples, len(X_test))

    selected_indices = np.random.choice(X_test.index, samples_used, replace=False)

    for idx in selected_indices:
        instance = X_test.loc[idx]
        current_pred = model.predict(instance.to_frame().T)[0]

        possible_classes = [0,1,2]
        desired_classes = [cls for cls in possible_classes if cls != current_pred]

        for desired_class in desired_classes:
            counterfactuals = explainer.generate_counterfactuals(
                instance.to_frame().T,
                total_CFs=cfs_per_instance,
                desired_class=desired_class,
                verbose=False
            )

            cf_df = counterfactuals.cf_examples_list[0].final_cfs_df.drop(columns=[target_col])

            for _, cf_instance in cf_df.iterrows():
                for feature in X_test.columns:
                    if cf_instance[feature] != instance[feature]:
                        feature_change_counts[feature] += 1

    feature_change_summary = pd.DataFrame.from_dict(feature_change_counts, orient='index', columns=['change_count'])
    total_possible_changes = samples_used * cfs_per_instance * 2  # two alternative classes per instance
    feature_change_summary['change_frequency'] = feature_change_summary['change_count'] / total_possible_changes
    feature_change_summary.sort_values('change_frequency', ascending=False, inplace=True)

    print(feature_change_summary)

    results_summary['RandomForest'] = feature_change_summary

    return results_summary


# --- Main app ---
def main():
    st.set_page_config(page_title="Stress Prediction Dashboard", layout="wide")
    st.title("🧠 What’s Your Stress Risk? Let’s Break It Down")

    input_df = user_input_features()
    processed_input = preprocess_input(input_df)

    if st.button("🔍 Predict Stress"):
      st.write('Your Input:- ')
      st.write(input_df)
        # Note: model expects input to be preprocessed/encoded like during training
        # Assuming model can handle categorical inputs directly or preprocessing is handled
      try:
        prediction = modelRandomForest.predict(processed_input)[0]
        prediction_proba = modelRandomForest.predict_proba(processed_input)[0]

        predicted_label = class_labels[prediction]

        if predicted_label == "Yes":
          st.error("🚨 You have a **high chance** of experiencing growing stress. It's recommended to seek support, rest, and possibly consult a mental health professional.")
        elif predicted_label == "No":
          st.success("✅ You have a **low chance** of experiencing growing stress. Keep maintaining a healthy routine and mental well-being!")
        elif predicted_label == "Maybe":
          st.warning("⚠️ There is a **moderate chance** of growing stress. Stay aware of changes in mood or habits and consider checking in with yourself or someone you trust.")
        else:
          st.info(f"Prediction: {predicted_label}")
        st.write("Probabilities:", dict(zip(class_labels, prediction_proba)))

        fig = plot_prediction_probabilities(prediction_proba, class_labels)
        st.plotly_chart(fig)
      except Exception as e:
        st.error(f"Prediction failed: {e}")

      st.subheader("📊 Counterfactual Analysis: How to Improve")
      st.info("Which features, if changed, could reduce your stress prediction?")

    # all_cf_df = generate_counterfactual_for_instance(
    #   model=modelRandomForest,
    #   input_instance=preprocess_input(input_df),
    #   dataset_path="MentalHealthDataEncoded.csv", 
    #   target_col="Growing_Stress",
    #   cfs_per_class=2
    # )
    # st.dataframe(all_cf_df)

      feature_summaries = multiclass_counterfactual_analysis_all_models(
        model=modelRandomForest,
        dataset_path='MentalHealthDataEncoded.csv',
        X_test=processed_input,
        num_samples=50,
        cfs_per_instance=2,
        target_col='Growing_Stress'
      )

      st.dataframe(feature_summaries['RandomForest'])

    # df = pd.read_csv(processed_input)
    # st.dataframe(processed_input)

if __name__ == '__main__':
    main()
