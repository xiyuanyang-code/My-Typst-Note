# File: features/steps/common_steps.py
from behave import given

@given('the user has entered the login page by clicking "Sign in" from the homepage')
def step_enter_login_page(context):
    context.current_page = "login"

@given('the user has entered the registration page by clicking "Register for free" from the homepage')
def step_enter_register_page(context):
    context.current_page = "register"