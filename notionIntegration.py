from os import getenv
import json
import requests
import threading
from time import time
from datetime import datetime
from tqdm.auto import tqdm as tqdm_auto

BASE_URL = "https://api.notion.com/v1"

def char_bar(value, filled="▓", empty="░"):
    """
    Parameters
    ----------
    value: int, required. Completion amount between 0 and 1.
    filled: str, required. The character for a completed progress.
        [default: "▓"].
    empty: str, required. The character for incomplete progress.
        [default: "░"]
    """
    f = int(value*10)
    e = 10 - f
    return (f * filled) + (e * empty)

class tqdm_notion(tqdm_auto):
    """
    Standard `tqdm.auto.tqdm` but also sends updates to a Notion Database.
    """


    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        secret: str, required. Notion Integration secret
            [default: ${TQDM_NOTION_SECRET}].
        database_id: str, required. Notion database id.
            [default: ${TQDM_NOTION_DATABASE_ID}].
        unique_property: str, required. The unique property for this database.
            [default: "Name"]
        page_title: str, required. Title of the database page.
            [default: {datetime.now().strftime("%d/%m, %H:%M")}]
        progress_property: str, required. Database property name.
            [default: "Progress"]
        update_interval_secs: int, required. Minimum time between calls to the API in seconds.
            [default: 1]
        See `tqdm.auto.tqdm.__init__` for other parameters.
        """
        if not kwargs.get('disable'):
            kwargs = kwargs.copy()

            # Get all the kwargs for later use
            self.secret = kwargs.pop('secret', getenv("TQDM_NOTION_SECRET"))
            self.database_id = kwargs.pop('database_id', getenv("TQDM_NOTION_DATABASE_ID"))
            self.unique_property = kwargs.pop('unique_property', "Name")
            self.page_title = kwargs.pop('page_title', datetime.now().strftime("%d-%m, %H:%M"))
            self.progress_property = kwargs.pop('progress_property', "Progress")
            self.complete_char = kwargs.pop('complete_char', "▓")
            self.incomplete_char = kwargs.pop('incomplete_char', "░")
            self.start_time_property = kwargs.pop('start_time_property', "Start")
            self.update_interval_secs = kwargs.pop('update_interval_secs', 1)

            self.last_update_time = time()
            self.loading = True

            # Now we've popped the kwargs we can call super
            super(tqdm_notion, self).__init__(*args, **kwargs)

            self.headers = {
                "Authorization": f"Bearer {self.secret}", 
                "Content-Type": "application/json",
                "Notion-Version": "2021-08-16"
                }
            
            page = requests.post(f"{BASE_URL}/pages", headers=self.headers, data=json.dumps({
                "parent": {
                    "database_id": self.database_id
                },
                "properties": {
                    self.unique_property: {
                        "title": [
                            {
                                "type": "text",
                                "text": {
                                    "content": self.page_title
                                }
                            }
                        ]
                    },
                    self.progress_property: {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"{self.bar} {self.percent_complete}%"
                            }
                        }
                    ]
                }
                }
            }))
            self.page_id = page.json()["id"]
            self.loading = False

        super(tqdm_notion, self).__init__(*args, **kwargs)
        
    @property
    def percent_complete(self):
        return int((self.n/self.total)*100)

    @property
    def bar(self):
        return char_bar(self.n/self.total, filled=self.complete_char, empty=self.incomplete_char)

    @property
    def can_update(self):
        interval_elapsed = (
            self.last_update_time is None
            or (time() - self.last_update_time) > self.update_interval_secs
        )
        return not self.loading and interval_elapsed

    def update_page(self, force = False):
        """
        This method should be called in its own thread to prevent blocking on the request.
        
        Force is used to force update when progress reaches 100%.
        """
        if not self.can_update and not force:
            return

        bar = self.bar

        self.loading = True
        page = requests.patch(f"{BASE_URL}/pages/{self.page_id}", headers=self.headers, data=json.dumps({
            "properties": {
                self.progress_property: {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"{bar} {self.percent_complete}%"
                            }
                        }
                    ]
                }
            }
        }))
        self.last_update_time = time()
        self.loading = False
    
    def display(self, **kwargs):
        super(tqdm_notion, self).display(**kwargs)

        # Start a thread to do the updating so we aren't waiting around for Notion.
        t = threading.Thread(
            name="update_page", target=self.update_page
        )
        t.setDaemon(True)
        t.start()

    def clear(self, *args, **kwargs):
        super(tqdm_notion, self).clear(*args, **kwargs)
        if not self.disable:
            # self.dio.write("")
            pass

    def close(self):
        self.update_page(force=True)
        super(tqdm_notion, self).close()