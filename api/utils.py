import pandas as pd

feature_order = [
  'duration', 'credit_amount', 'monthly_payment', 'age',
  'installment_rate', 'residence_since', 'existing_credits', 'num_dependents',
  'financial_stability', 'employment_points', 'overall_stability', 'purpose_points',
  'credit_history_A31', 'credit_history_A32', 'credit_history_A33', 'credit_history_A34',
  'personal_status_A92', 'personal_status_A93', 'personal_status_A94',
  'other_parties_A102', 'other_parties_A103',
  'property_A122', 'property_A123', 'property_A124',
  'other_payment_plans_A142', 'other_payment_plans_A143',
  'housing_A152', 'housing_A153',
  'job_A172', 'job_A173', 'job_A174',
  'telephone_A192', 'foreign_worker_A202'
]

checking_map = {  'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3}

savings_map = {'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65': 2}

employment_map = {'A71': 0, 'A72': 0, 'A73': 1, 'A74': 2, 'A75': 2}

purpose_map = {
  **dict.fromkeys(['A40', 'A410', 'A46'], 0),
  **dict.fromkeys(['A42', 'A44', 'A45', 'A49'], 1),
  **dict.fromkeys(['A41', 'A43'], 2),
  'A48': 3
}

def preprocess_input(data: dict) -> pd.DataFrame:
  row = {}
  checking_points = checking_map[data['checking_status']]
  savings_points = savings_map[data['savings_status']]
  employment_points = employment_map[data['employment_status']]
  purpose_points = purpose_map[data['purpose']]

  row['duration'] = data['duration']
  row['credit_amount'] = data['credit_amount']
  row['installment_rate'] = data['installment_rate']
  row['residence_since'] = data['residence_since']
  row['age'] = data['age']
  row['existing_credits'] = data['existing_credits']
  row['num_dependents'] = data['num_dependents']

  row['monthly_payment'] = data['credit_amount'] / data['duration']
  row['financial_stability'] = checking_points + savings_points
  row['employment_points'] = employment_points
  row['overall_stability'] = row['financial_stability'] + employment_points
  row['purpose_points'] = purpose_points
  
  for col in ['A31', 'A32', 'A33', 'A34']:
    row[f'credit_history_{col}'] = 1 if data['credit_history'] == col else 0

  for col in ['A92', 'A93', 'A94']:
    row[f'personal_status_{col}'] = 1 if data['personal_status'] == col else 0

  for col in ['A102', 'A103']:
    row[f'other_parties_{col}'] = 1 if data['other_parties'] == col else 0

  for col in ['A122', 'A123', 'A124']:
    row[f'property_{col}'] = 1 if data['property_type'] == col else 0

  for col in ['A142', 'A143']:
    row[f'other_payment_plans_{col}'] = 1 if data['other_payment_plans'] == col else 0

  for col in ['A152', 'A153']:
    row[f'housing_{col}'] = 1 if data['housing'] == col else 0

  for col in ['A172', 'A173', 'A174']:
    row[f'job_{col}'] = 1 if data['job'] == col else 0

  row['telephone_A192'] = 1 if data['telephone'] == 'A192' else 0

  row['foreign_worker_A202'] = 1 if data['foreign_worker'] == 'A202' else 0

  return pd.DataFrame([row])[feature_order]