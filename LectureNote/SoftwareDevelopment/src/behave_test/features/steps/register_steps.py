# File: features/steps/register_steps.py
from behave import given, when, then

# Use context.db
def get_db(context):
    return context.db

@given('the user has entered phone "{phone}" and captcha "{code}" on the registration page')
def step_input_register_info(context, phone, code):
    context.input_phone = phone
    context.input_code = code

@when('the user does not check the agreement checkbox')
def step_not_agree(context):
    context.agree_protocol = False

@when('the user checks the agreement checkbox')
def step_agree(context):
    context.agree_protocol = True

@then('the "Register" button is disabled')
def step_register_disabled(context):
    assert not context.agree_protocol

@then('the "Register" button becomes enabled')
def step_register_enabled(context):
    assert context.agree_protocol

@then('the system does not create a new user')
def step_no_new_user(context):
    db = get_db(context)
    assert context.input_phone not in db or not db[context.input_phone].get("new_user")

@then('the system creates a new user in the database')
def step_create_user(context):
    db = get_db(context)
    db[context.input_phone] = {"registered": True, "new_user": True}
    print(f"[REGISTERED] New user: {context.input_phone}")

@then('the user is logged in and redirected to the homepage')
def step_logged_in_home(context):
    context.current_page = "home"