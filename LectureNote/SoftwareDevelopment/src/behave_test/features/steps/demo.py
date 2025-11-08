from behave import given, when, then

@given('we have behave installed')
def step_given_behave_installed(context):
    pass

@when('we implement a test')
def step_when_implement_test(context):
    context.executed = True

@then('behave will test it for us!')
def step_then_behave_tests_it(context):
    # 断言测试被执行
    assert hasattr(context, 'executed'), "Test was not implemented!"
    print("Behave 测试成功！")