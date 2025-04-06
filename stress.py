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

def decode_one_hot_to_categorical(df_counterfactual_one_hot):
    """
    Decodes a one-hot encoded dataset back to categorical values based on the column names.
    Handles column names with multiple underscores.

    Parameters:
    df_counterfactual_one_hot (pd.DataFrame): The one-hot encoded counterfactual dataset.

    Returns:
    pd.DataFrame: A DataFrame with the counterfactual values decoded back to their original categorical format.
    """
    
    # Initialize an empty DataFrame to store decoded counterfactual values
    decoded_df = pd.DataFrame(index=df_counterfactual_one_hot.index)
    
    # Iterate over columns to decode based on the pattern
    for col in df_counterfactual_one_hot.columns:
        # Try to extract the category name based on the naming convention
        # We will find the first occurrence of an underscore after the first part of the column name
        category_name = '_'.join(col.split('_')[:-1])  # Take everything except the last part as category name
        
        if category_name not in decoded_df.columns:
            # Get the one-hot encoded columns related to the same original category
            one_hot_cols = [col for col in df_counterfactual_one_hot.columns if col.startswith(category_name)]
            
            if one_hot_cols:
                # Find the index of the '1' (encoded value) for each row
                decoded_values = df_counterfactual_one_hot[one_hot_cols].idxmax(axis=1)
                # Clean up the column name to retrieve the actual category value
                decoded_values = decoded_values.str.replace(f"{category_name}_", "")
                decoded_df[category_name] = decoded_values
    
    return decoded_df

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

  original_pred = model.predict(input_instance.loc[0].to_frame().T,)[0]
  print(original_pred)
  possible_classes = [0, 1, 2]
  desired_classes = [cls for cls in possible_classes if cls != original_pred]

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

  all_cf_df = pd.concat(all_counterfactuals, ignore_index=True)

  return all_cf_df

  # return all_cf_df


def multiclass_counterfactual_analysis_all_models(model, dataset_path, X_test, num_samples=50, cfs_per_instance=2, target_col='Growing_Stress'):
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

def aggregate_feature_summaries(feature_summaries):
    aggregated_dict = {}

    for model_name, summary_df in feature_summaries.items():
        aggregated = {}

        for feature in summary_df.index:
            # Try splitting only the last part after last underscore
            if '_' in feature:
                base = '_'.join(feature.split('_')[:-1])
            else:
                base = feature

            if base in aggregated:
                aggregated[base] += summary_df.loc[feature, 'change_count']
            else:
                aggregated[base] = summary_df.loc[feature, 'change_count']

        # Convert to DataFrame
        agg_df = pd.DataFrame.from_dict(aggregated, orient='index', columns=['aggregated_change_count'])
        total = agg_df['aggregated_change_count'].sum()
        agg_df['aggregated_change_frequency'] = agg_df['aggregated_change_count'] / total
        agg_df.sort_values('aggregated_change_frequency', ascending=False, inplace=True)

        aggregated_dict[model_name] = agg_df

    return aggregated_dict


def plot_feature_summary_for_random_forest(feature_summaries):
    """
    Creates a bar plot for aggregated count and frequency for RandomForest using Plotly.
    
    Parameters:
    feature_summaries (dict or pd.DataFrame): The aggregated feature summaries.
    """
    
    # Aggregate the feature summaries
    aggregated_summary = aggregate_feature_summaries(feature_summaries)
    
    # Extract the RandomForest data (assuming it's in a dictionary or DataFrame)
    random_forest_data = aggregated_summary['RandomForest']
    
    # Create a DataFrame to better handle plotting
    df = pd.DataFrame(random_forest_data)
    
    # Check if 'count' and 'frequency' columns exist
    if 'aggregated_change_count' in df.columns and 'aggregated_change_frequency' in df.columns:
        # Set the x-axis as feature names or indices
        features = df.index
        
        # Create the bar trace for 'count'
        count_trace = go.Bar(
            x=features,
            y=df['aggregated_change_count'],
            name='Count',
            opacity=0.7
        )
        
        # Create the line trace for 'frequency'
        frequency_trace = go.Scatter(
            x=features,
            y=df['aggregated_change_frequency'],
            name='Frequency',
            mode='lines+markers',
            line=dict(color='red', width=2)
        )
        
        # Create the layout with labels and title
        layout = go.Layout(
            title='Aggregated Count and Frequency for RandomForest',
            xaxis=dict(title='Features', tickangle=45),
            yaxis=dict(title='Values'),
            barmode='group'
        )
        
        # Create the figure and add traces
        fig = go.Figure(data=[count_trace, frequency_trace], layout=layout)
        
        # Show the plot
        st.plotly_chart(fig)
    else:
        print("The expected 'count' and 'frequency' columns were not found in the RandomForest data.")


# --- Main app ---
def main():
    st.set_page_config(page_title="Stress Prediction Dashboard", layout="wide")
    st.title("🧠 What’s Your Stress Risk? Let’s Break It Down")

    input_df = user_input_features()
    processed_input = preprocess_input(input_df)

    if st.button("🔍 Predict Stress"):
      st.write('Your Input:- ')
      st.write(input_df)
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
      st.info("Which features, if changed, could impact your stress outcome?")

      # all_cf_df = generate_counterfactual_for_instance(
      #   model=modelRandomForest,
      #   input_instance=preprocess_input(input_df),
      #   dataset_path="MentalHealthDataEncoded.csv", 
      #   target_col="Growing_Stress",
      #   cfs_per_class=2,
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

      # st.dataframe(feature_summaries['RandomForest'])
      st.dataframe(aggregate_feature_summaries(feature_summaries)['RandomForest'])

      plot_feature_summary_for_random_forest(feature_summaries)

      # df = pd.read_csv(processed_input)
      # st.dataframe(processed_input)

if __name__ == '__main__':
    main()
