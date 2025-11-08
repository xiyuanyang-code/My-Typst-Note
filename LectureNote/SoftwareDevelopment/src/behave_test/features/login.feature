# language: en
# File: features/login.feature

Feature: User Login

  Background:
    Given the user has entered the login page by clicking "Sign in" from the homepage
    And the login page contains phone number field, captcha field, "Get Captcha" button, and "Login" button

  Scenario: Invalid phone number when requesting captcha
    When the user enters an invalid phone number "123"
    And clicks the "Get Captcha" button
    Then the system does not send a captcha
    And the page shows the message "Please enter a valid phone number"

  Scenario: Successfully request captcha
    When the user enters a valid phone number "13800138000"
    And clicks the "Get Captcha" button
    Then the system generates a 6-digit captcha for the phone and prints it to the console
    And the "Get Captcha" button starts a 60-second countdown and becomes disabled
    And the database records the phone, captcha, and 60-second expiry

  Scenario: Login with unregistered phone
    Given the phone number "13800138999" is not registered
    When the user enters that phone and a correct captcha "123456"
    And clicks the "Login" button
    Then the system denies login
    And the page shows the message "This phone is not registered. Please register first."

  Scenario: Login with wrong captcha
    Given the phone number "13800138000" is registered
    When the user enters that phone and an incorrect captcha "000000"
    And clicks the "Login" button
    Then the system denies login
    And the page shows the message "Incorrect captcha"

  Scenario: Successful login
    Given the phone number "13800138000" is registered
    And the system has generated captcha "123456" for it
    When the user enters that phone and captcha "123456"
    And clicks the "Login" button
    Then the system authenticates successfully
    And the page shows the message "Login successful"
    And the page automatically redirects to the homepage