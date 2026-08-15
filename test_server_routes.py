from unittest import TestCase

from fastapi.responses import RedirectResponse

from server import app, root


class ServerRouteTests(TestCase):
    def test_root_redirects_to_api_docs(self):
        response = root()

        self.assertIsInstance(response, RedirectResponse)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/docs")
        self.assertTrue(any(route.path == "/" for route in app.routes))