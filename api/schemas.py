from pydantic import BaseModel, validator

class LoanApplication(BaseModel):
  duration: int
  credit_amount: float
  installment_rate: int
  residence_since: int
  age: int
  existing_credits: int
  num_dependents: int
  credit_history: str
  personal_status: str
  other_parties: str
  property_type: str
  other_payment_plans: str
  housing: str
  job: str
  telephone: str
  foreign_worker: str
  # engineered features - user provides raw values, we compute
  checking_status: str
  savings_status: str
  employment_status: str
  purpose: str

  @validator('checking_status')
  def valid_checking(cls, v):
    if v not in ['A11', 'A12', 'A13', 'A14']:
      raise ValueError('Invalid checking_status code')
    return v
  @validator('savings_status')
  def valid_savings(cls, v):
    if v not in ['A61', 'A62', 'A63', 'A64', 'A65']:
      raise ValueError('Invalid savings_status code')
    return v
  @validator('employment_status')
  def valid_employment(cls, v):
    if v not in ['A71', 'A72', 'A73', 'A74', 'A75']:
      raise ValueError('Invalid employment_status code')
    return v
  @validator('purpose')
  def valid_purpose(cls, v):
    if v not in ['A40', 'A410', 'A46', 'A42', 'A44', 'A45', 'A49','A41', 'A43','A48']:
      raise ValueError('Invalid purpose')
    return v