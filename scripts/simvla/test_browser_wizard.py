"""Opt-in real Chromium smoke of the wizard DOM/HTTP flow, without Isaac or scene writes.

Install playwright and its Chromium browser, then set SIMVLA_BROWSER_SMOKE=1.
This does not validate the GPU director, 3D preview, or generated-scene physics.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SIMVLA_BROWSER_SMOKE") != "1",
                                reason="explicit browser smoke only")


@pytest.mark.parametrize("robot", ["anubis", "rby1", "aiworker"])
def test_browser_setup_submission_and_live_step_transition(tmp_path, robot):
    from playwright.sync_api import sync_playwright, expect
    from kitchen_wizard import WizardServer, form_payload, render_form_page, render_choice_page

    server = WizardServer(port=0)
    errors = []
    try:
        server.set_setup_context(mesh_files=[])
        with sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(server.url)
                expect(page.get_by_role("heading", name="Starting…")).to_be_visible()
                server.publish("form", render_form_page(form_payload(str(tmp_path), {"robot": robot})))
                expect(page.locator("#num")).to_be_visible(timeout=10000)
                page.locator("#clear").click()
                page.locator("#num").fill("invalid")
                page.locator("#generate").click()
                expect(page.locator("#err")).to_contain_text("non-negative integer")
                assert not server._answered.is_set()
                page.locator("#num").fill("99101")
                with page.expect_response(lambda response: response.url.endswith("/setup")) as submitted:
                    page.locator("#generate").click()
                assert submitted.value.status == 200
                assert server._answered.wait(5), "browser never submitted a valid setup"
                answer = server.wait_answer()
                assert answer["kitchen_num"] == 99101
                assert answer["robot"] == robot
                assert answer["objects"] == []

                options = [{"id": "review", "label": "Review generated scene"},
                           {"id": "cancel", "label": "Cancel"}]
                server.publish("choice", render_choice_page(title="Next step", text="Smoke only",
                                                             options=options), options=[option["id"] for option in options])
                expect(page.get_by_role("button", name="Review generated scene")).to_be_visible(timeout=10000)
                with page.expect_response(lambda response: response.url.endswith("/answer")):
                    page.get_by_role("button", name="Review generated scene").click()
                assert server._answered.wait(5)
                assert server.wait_answer()["id"] == "review"
                assert not errors, errors
            finally:
                browser.close()
    finally:
        server.shutdown()
