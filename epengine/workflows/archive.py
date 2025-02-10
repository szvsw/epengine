"""A module for copying the contents of another bucket to a new bucket."""

import asyncio

import boto3
from hatchet_sdk import Context
from pydantic import BaseModel

from epengine.hatchet import hatchet

s3 = boto3.client(
    "s3",
)


class CheckFolder(BaseModel):
    """A class for checking the contents of a folder in a bucket.

    Splits between folders and files.
    """

    source_bucket: str
    source_folder: str

    destination_bucket: str
    destination_folder: str

    def check_folder(self) -> tuple[list[str], list[str]]:  # noqa: C901
        """Check the contents of a folder in a bucket.

        Returns a tuple of two lists:
        - The first list contains the subfolders in the folder.
        - The second list contains the files in the folder.
        """
        paginator = s3.get_paginator("list_objects_v2")

        subfolders = []
        files = []
        response = paginator.paginate(
            Bucket=self.source_bucket, Prefix=self.source_folder, Delimiter="/"
        )
        for page in response:
            for obj in page.get("CommonPrefixes", []):
                if "Prefix" not in obj:
                    continue
                prefix = obj["Prefix"]
                if prefix == self.source_folder:
                    continue
                if "by_ashrae_iecc_climate_zone" in prefix:
                    continue
                if "by_building_america_climate_zone" in prefix:
                    continue
                if "by_iso_rto_region" in prefix:
                    continue
                if "upgrade=" in prefix and "upgrade=0" not in prefix:
                    continue
                subfolders.append(prefix)
            for obj in page.get("Contents", []):
                if "Key" not in obj:
                    continue
                if obj["Key"].endswith("/"):
                    continue
                else:
                    files.append(obj["Key"])

        return subfolders, files


class FileToTransfer(BaseModel):
    """A class for copying a file to a new bucket.

    Copies a file to a new bucket.
    """

    source_file_key: str
    source_bucket: str
    destination_bucket: str
    destination_folder: str

    @property
    def destination_file_key(self) -> str:
        """The destination file key.

        Returns the destination file key.
        """
        return f"{self.destination_folder}/{self.source_file_key}"

    def copy_file(self):
        """Copy a file to a new bucket.

        Copies a file to a new bucket.
        """
        s3.copy_object(
            Bucket=self.destination_bucket,
            CopySource=f"{self.source_bucket}/{self.source_file_key}",
            Key=self.destination_file_key,
            # update the storage class to infrequent access
            # StorageClass="STANDARD_IA",
            # update the storage class to deep glacier
            StorageClass="DEEP_ARCHIVE",
        )


@hatchet.workflow(
    name="archive_folder",
    timeout="1000m",
    schedule_timeout="1000m",
)
class ArchiveFolder:
    """A workflow for archiving a folder."""

    @hatchet.step(name="archive_folder", timeout="100m", retries=2)
    async def archive_folder(self, context: Context):
        """Archive a folder.

        Archives a folder.
        """
        data = context.workflow_input()
        check_folder = CheckFolder(**data)
        subfolders, files = check_folder.check_folder()

        promises = []
        for folder in subfolders:
            payload = CheckFolder(
                source_bucket=data["source_bucket"],
                source_folder=folder,
                destination_bucket=data["destination_bucket"],
                destination_folder=data["destination_folder"],
            )
            task = context.aio.spawn_workflow(
                "archive_folder",
                payload.model_dump(mode="json"),
            )
            promises.append(task)
        await asyncio.gather(*promises, return_exceptions=True)
        for i in range(0, len(files), 1000):
            if (i // 1000) % 10 == 0:
                print(f"step {i // 1000} of {len(files) // 1000}")
            batch = files[i : min(i + 1000, len(files))]
            promises = []
            for file in batch:
                payload = FileToTransfer(
                    source_file_key=file,
                    source_bucket=data["source_bucket"],
                    destination_bucket=data["destination_bucket"],
                    destination_folder=data["destination_folder"],
                )
                task = context.aio.spawn_workflow(
                    "archive_file",
                    payload.model_dump(mode="json"),
                )
                promises.append(task)

            await asyncio.gather(*promises, return_exceptions=True)

        return {"message": "success"}


@hatchet.workflow(
    name="archive_file",
    timeout="1000m",
    schedule_timeout="1000m",
)
class ArchiveFile:
    """A workflow for archiving a file."""

    @hatchet.step(name="archive_file", timeout="100m", retries=2)
    def archive_file(self, context: Context):
        """Archive a file.

        Archives a file.
        """
        data = context.workflow_input()
        file = FileToTransfer(**data)
        file.copy_file()
        return {"message": "success"}


if __name__ == "__main__":
    check_folder = CheckFolder(
        source_bucket="oedi-data-lake",
        source_folder="nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_tmy3_release_2/",
        destination_bucket="mit-sdl-archive",
        destination_folder="nrel-archive",
    )

    from hatchet_sdk import new_client

    client = new_client()
    client.admin.run_workflow(
        "archive_folder",
        check_folder.model_dump(mode="json"),
    )
    # subfolders, files = check_folder.check_folder()
    # print(subfolders)
    # print(len(files))
    # for folder in subfolders:
    #     new_folder = CheckFolder(
    #         source_bucket="oedi-data-lake",
    #         source_folder=folder,
    #         destination_bucket="ml-for-bem",
    #         destination_folder="nrel-archive",
    #     )
    #     subsubfolders, subfiles = new_folder.check_folder()
    #     print(len(subfiles))
    #     for subsubfolder in subsubfolders:
    #         if subsubfolder in subfolders:
    #             raise ValueError("err", subsubfolder)
    #         if "model_and_schedule" in subsubfolder:
    #             new_folder = CheckFolder(
    #                 source_bucket="oedi-data-lake",
    #                 source_folder=subsubfolder,
    #                 destination_bucket="ml-for-bem",
    #                 destination_folder="nrel-archive",
    #             )
    #             subsubsubfolders, _ = new_folder.check_folder()
    #             print(len(_))
    #             for subsubsubfolder in subsubsubfolders:
    #                 if subsubsubfolder in subfolders:
    #                     raise ValueError("err", subsubsubfolder)
