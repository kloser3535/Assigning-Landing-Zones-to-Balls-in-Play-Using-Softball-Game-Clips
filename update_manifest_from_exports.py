import pandas as pd
import os


MANIFEST_CSV = r"C:\Users\klose\Downloads\ScreenTrackProject\manifest.csv"

manifest = pd.read_csv(MANIFEST_CSV)

# Add columns if they do not exist
if "zone" not in manifest.columns:
    manifest["zone"] = ""

if "zone_assigned" not in manifest.columns:
    manifest["zone_assigned"] = False

if "pipeline_status" not in manifest.columns:
    manifest["pipeline_status"] = ""

manifest["zone"] = manifest["zone"].astype("string")
manifest["pipeline_status"] = manifest["pipeline_status"].astype("string")

success = 0
unassigned = 0
no_calibration = 0
missing_export = 0


for index, row in manifest.iterrows():

    export_path = row["export_csv"]

    export_row = int(row["export_row"])


    if not os.path.exists(export_path):

        print(
            "Missing export:",
            export_path
        )

        manifest.loc[index, "pipeline_status"] = "MISSING_EXPORT"
        manifest.loc[index, "zone_assigned"] = False

        missing_export += 1
        continue


    try:

        export_df = pd.read_csv(export_path)


        # -----------------------------------
        # No zone column means calibration failed
        # -----------------------------------

        # -----------------------------------
        # Find zone column
        # -----------------------------------

        if "zone" in export_df.columns:

            zone_column = "zone"

        elif "BB_zone" in export_df.columns:

            zone_column = "BB_zone"

        else:

            # Neither column exists = no calibration result

            manifest.loc[index, "zone"] = ""

            manifest.loc[index, "zone_assigned"] = False

            manifest.loc[index, "pipeline_status"] = (
                "NO_CALIBRATION_FRAME"
            )

            no_calibration += 1

            continue

        zone = export_df.loc[
            export_row,
            zone_column
        ]


        # Clean zone formatting

        if pd.notna(zone):

            if isinstance(zone, float) and zone.is_integer():

                zone = str(int(zone))

            else:

                zone = str(zone).strip()

        if pd.notna(zone):

            # convert numbers like 4.0 -> 4
            if isinstance(zone, float) and zone.is_integer():

                zone = str(int(zone))

            else:

                zone = str(zone).strip()


    except Exception as e:

        print(
            "Failed:",
            export_path,
            e
        )

        manifest.loc[index, "zone"] = ""
        manifest.loc[index, "zone_assigned"] = False
        manifest.loc[index, "pipeline_status"] = "ERROR"

        continue



    # -------------------------
    # Determine pipeline status
    # -------------------------

    # Blank zone
    if pd.isna(zone) or str(zone).strip() == "":

        manifest.loc[index, "zone"] = ""

        manifest.loc[index, "zone_assigned"] = False

        manifest.loc[index, "pipeline_status"] = (
            "NO_CALIBRATION_FRAME"
        )

        no_calibration += 1


    # Calibration succeeded but zone assignment failed
    elif str(zone).strip().upper() == "UNASSIGNED":

        manifest.loc[index, "zone"] = ""

        manifest.loc[index, "zone_assigned"] = False

        manifest.loc[index, "pipeline_status"] = (
            "UNASSIGNED"
        )

        unassigned += 1


    # Valid zone
    else:

        manifest.loc[index, "zone"] = zone

        manifest.loc[index, "zone_assigned"] = True

        manifest.loc[index, "pipeline_status"] = (
            "SUCCESS"
        )

        success += 1



print()
print("======================")
print("Manifest update complete")
print("======================")
print("Success:", success)
print("Unassigned:", unassigned)
print("No calibration:", no_calibration)
print("Missing export:", missing_export)

print("Pipeline Status Value Counts")
print(manifest["pipeline_status"].value_counts())


manifest.to_csv(
    MANIFEST_CSV,
    index=False
)
