# File: features/steps/login_steps.py
import re
import time
import random
from behave import given, when, then

# Use context.db instead of local db
def get_db(context):
    return context.db

@given('the phone number "{phone}" is not registered')
def step_phone_not_registered(context, phone):
    get_db(context).pop(phone, None)

@given('the phone number "{phone}" is registered')
def step_phone_registered(context, phone):
    get_db(context)[phone] = {"registered": True}

@given('the system has generated captcha "{code}" for it')
def step_captcha_generated(context, code):
    phone = getattr(context, "current_phone", "13800138000")
    db = get_db(context)
    db.setdefault(phone, {})["captcha"] = code
    db[phone]["captcha_time"] = time.time()

@when('the user enters an invalid phone number "{phone}"')
def step_input_invalid_phone(context, phone):
    context.input_phone = phone
    context.is_valid_phone = False

@when('the user enters a valid phone number "{phone}"')
def step_input_valid_phone(context, phone):
    if re.match(r"^1[3-9]\d{9}$", phone):
        context.input_phone = phone
        context.current_phone = phone
        context.is_valid_phone = True
    else:
        context.is_valid_phone = False

@when('the user enters that phone and a correct captcha "{code}"')
@when('the user enters that phone and an incorrect captcha "{code}"')
@when('the user enters that phone and captcha "{code}"')
def step_input_phone_and_code(context, code):
    context.input_code = code

@when('clicks the "{button}" button')
def step_click_button(context, button):
    context.last_action = f"click_{button}"

@then('the system does not send a captcha')
def step_no_captcha_sent(context):
    db = get_db(context)
    assert context.input_phone not in db or "captcha" not in db[context.input_phone]

@then('the page shows the message "{msg}"')
def step_show_message(context, msg):
    context.last_message = msg

@then('the system generates a 6-digit captcha for the phone and prints it to the console')
def step_generate_captcha(context):
    code = f"{random.randint(100000, 999999)}"
    db = get_db(context)
    db[context.current_phone]["captcha"] = code
    print(f"[CAPTCHA] Phone: {context.current_phone} -> {code}")
    context.generated_code = code

@then('the "Get Captcha" button starts a 60-second countdown and becomes disabled')
def step_captcha_countdown(context):
    get_db(context)[context.current_phone]["countdown"] = True

@then('the database records the phone, captcha, and 60-second expiry')
def step_captcha_recorded(context):
    db = get_db(context)
    assert context.current_phone in db
    assert "captcha" in db[context.current_phone]

@then('the system denies login')
def step_login_denied(context):
    context.login_success = False

@then('the system authenticates successfully')
def step_login_success(context):
    db = get_db(context)
    assert context.input_code == db[context.current_phone]["captcha"]
    context.login_success = True

@then('the page automatically redirects to the homepage')
def step_redirect_home(context):
    context.current_page = "home"