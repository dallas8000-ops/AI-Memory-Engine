from unittest import TestCase

from fastapi.responses import FileResponse

from server import app, root


class ServerRouteTests(TestCase):
    def test_root_serves_memory_console(self):
        response = root()

        self.assertIsInstance(response, FileResponse)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.path.name, "index.html")
        self.assertEqual(response.path.parent.name, "web")
        self.assertTrue(any(route.path == "/" for route in app.routes))