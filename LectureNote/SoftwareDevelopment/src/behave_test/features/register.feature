# language: en
# File: features/register.feature

Feature: User Registration

  Background:
    Given the user has entered the registration page by clicking "Register for free" from the homepage
    And the registration page contains phone number field, captcha field, "Get Captcha" button, agreement checkbox, and "Register" button

  Scenario: Invalid phone number when requesting captcha
    When the user enters an invalid phone number "123"
    And clicks the "Get Captcha" button
    Then the system does not send a captcha
    And the page shows the message "Please enter a valid phone number"

  Scenario: Successfully request captcha
    When the user enters a valid phone number "13800138999"
    And clicks the "Get Captcha" button
    Then the system generates a 6-digit captcha for the phone and prints it to the console
    And the "Get Captcha" button starts a 60-second countdown and becomes disabled
    And the database records the phone, captcha, and 60-second expiry

  Scenario: Register with already-registered phone
    Given the phone number "13800138000" is already registered
    When the user enters that phone and a correct captcha "123456"
    And checks the agreement checkbox
    And clicks the "Register" button
    Then the system does not create a new user
    And the page shows the message "This phone is already registered. Logging you in..."
    And the user is logged in and redirected to the homepage

  Scenario Outline: Register button state when agreement is not checked
    Given the user has entered phone "<phone>" and captcha "<code>" on the registration page
    When the user does not check the agreement checkbox
    Then the "Register" button is disabled
    When the user checks the agreement checkbox
    Then the "Register" button becomes enabled

    Examples:
      | phone         | code   |
      | 13800138999   | 123456 |

  Scenario: Successful registration
    Given the phone number "13800138999" is not registered
    And the system has generated captcha "123456" for it
    When the user enters that phone and captcha "123456"
    And checks the agreement checkbox
    And clicks the "Register" button
    Then the system creates a new user in the database
    And the page shows the message "Registration successful"
    And the user is logged in and automatically redirected to the homepage