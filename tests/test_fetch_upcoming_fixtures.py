import unittest
from unittest.mock import patch

import pandas as pd

import fetch_upcoming_fixtures as mod


class FetchUpcomingFixturesTests(unittest.TestCase):
    def test_fetch_upcoming_bl1_fixtures_falls_back_on_ssl_error(self):
        fake_event = {
            "date": "2026-06-10T18:00:00Z",
            "competitions": [
                {
                    "status": {"type": {"state": "scheduled"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Bayern Munich"}},
                        {"homeAway": "away", "team": {"displayName": "Dortmund"}},
                    ],
                }
            ],
        }

        with patch.object(mod, "FOOTBALL_DATA_KEY", "test-key"), \
             patch.object(mod, "_request_matches", side_effect=mod.requests.exceptions.SSLError("boom")), \
             patch.object(mod, "_fetch_from_espn", return_value=[fake_event]) as fallback_mock:
            df = mod.fetch_upcoming_bl1_fixtures()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["HomeTeam"], "Bayern Munich")
        self.assertEqual(df.iloc[0]["AwayTeam"], "Dortmund")
        fallback_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
