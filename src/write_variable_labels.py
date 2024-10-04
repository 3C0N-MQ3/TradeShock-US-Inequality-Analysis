import os
import pandas as pd
from toolz import pipe


def write_variable_labels():
    mainp = os.path.join(".", "data")

    # Ensure the data folder exists
    os.makedirs(mainp, exist_ok=True)

    var = pipe(
        os.path.join(mainp, "usa_00137.dta"),
        lambda x: pd.read_stata(x, iterator=True),  # Load the full dataset
        lambda x: x.variable_labels(),  # Extract variable labels
        lambda x: pd.DataFrame(
            list(x.items()), columns=["Variable", "Label"]
        ),  # Create DataFrame
    )

    # Save the DataFrame to CSV
    var.to_csv(os.path.join(mainp, "variable_labels.csv"), index=False)

    print("Variable labels have been saved to 'variable_labels.csv'")
