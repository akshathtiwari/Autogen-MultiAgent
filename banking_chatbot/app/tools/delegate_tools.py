# app/tools/delegate_tools.py

from autogen_core.tools import FunctionTool

#
# Each "transfer_to_*" function returns a string
# that identifies the topic type of the target domain agent.
#

def transfer_to_retail_banking() -> str:
    return "RetailBanking"

def transfer_to_corporate_banking() -> str:
    return "CorporateBanking"

def transfer_to_investment_banking() -> str:
    return "InvestmentBanking"

def transfer_to_wealth_management() -> str:
    return "WealthManagement"

def transfer_to_risk_management() -> str:
    return "RiskManagement"

def transfer_to_insurance() -> str:
    return "Insurance"

def transfer_to_it_ops() -> str:
    return "ITOps"

def transfer_to_payments() -> str:
    return "Payments"

def transfer_to_capital_treasury() -> str:
    return "CapitalTreasury"

def transfer_to_analytics() -> str:
    return "Analytics"

# Wrap them as Tools so an AI agent can "call" them
transfer_to_retail_banking_tool = FunctionTool(
    transfer_to_retail_banking,
    description="Call this to route the user to RetailBankingAgent."
)
transfer_to_corporate_banking_tool = FunctionTool(
    transfer_to_corporate_banking,
    description="Call this to route the user to CorporateBusinessBankingAgent."
)
transfer_to_investment_banking_tool = FunctionTool(
    transfer_to_investment_banking,
    description="Call this to route the user to InvestmentBankingAgent."
)
transfer_to_wealth_management_tool = FunctionTool(
    transfer_to_wealth_management,
    description="Call this to route the user to WealthManagementAgent."
)
transfer_to_risk_management_tool = FunctionTool(
    transfer_to_risk_management,
    description="Call this to route the user to RiskManagementAgent."
)
transfer_to_insurance_tool = FunctionTool(
    transfer_to_insurance,
    description="Call this to route the user to InsuranceAgent."
)
transfer_to_it_ops_tool = FunctionTool(
    transfer_to_it_ops,
    description="Call this to route the user to ITOpsAgent."
)
transfer_to_payments_tool = FunctionTool(
    transfer_to_payments,
    description="Call this to route the user to PaymentsAgent."
)
transfer_to_capital_treasury_tool = FunctionTool(
    transfer_to_capital_treasury,
    description="Call this to route the user to CapitalTreasuryAgent."
)
transfer_to_analytics_tool = FunctionTool(
    transfer_to_analytics,
    description="Call this to route the user to AnalyticsAgent."
)
