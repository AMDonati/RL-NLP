from __future__ import print_function

import argparse
import os.path
import pickle

import pandas as pd
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']


def main(args):
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    # Call the Drive v3 API
    id_parent_folder = args.id_folder

    page_token = None
    data = []
    i = 0
    while True:
        i += 1
        print("Iteration {}".format(i))
        response = service.files().list(q="'{}' in parents".format(id_parent_folder),
                                        spaces='drive',
                                        fields='nextPageToken, files(id, name)',
                                        pageToken=page_token).execute()
        for file in response.get('files', []):
            id_coco = file.get("name").split("_")[-1][:-4]
            data.append([id_coco, file.get('id')])

        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

    df = pd.DataFrame(data=data, columns=["id_image", "id_google"])
    df.set_index("id_image", inplace=True)
    df.to_csv("data/drive/image_ids_{}.csv".format(args.name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-id_folder", type=str, default="1J1K5aJAQ9RTmLmGCd7Qnnvuj03HJ3MPJ", help="id of the folder")
    parser.add_argument("-name", type=str, help="id of the folder")
    args = parser.parse_args()

    main(args)
