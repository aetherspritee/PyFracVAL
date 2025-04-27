"""
Pydantic models for simulation configuration and results data structure.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class SimulationParameters(BaseModel):
    """Input parameters for a FracVAL simulation run.

    Used for validation and type hinting of the simulation configuration.

    Attributes
    ----------
    N : int
        Target number of primary particles.
    Df : float
        Target fractal dimension.
    kf : float
        Target fractal prefactor.
    rp_g : float
        Geometric mean radius of primary particles.
    rp_gstd : float
        Geometric standard deviation of radii (must be >= 1.0).
    tol_ov : float
        Overlap tolerance (must be > 0.0).
    n_subcl_percentage : float
        Target fraction for PCA subcluster size (0.0 < perc <= 0.5).
    ext_case : int
        CCA sticking ext_case (0 or 1).
    seed : int | None
        Random seed used for generation (optional).
    """

    N: int = Field(..., description="Target number of primary particles.")
    Df: float = Field(..., description="Target fractal dimension.")
    kf: float = Field(..., description="Target fractal prefactor.")
    rp_g: float = Field(..., description="Geometric mean radius of primary particles.")
    rp_gstd: float = Field(
        ..., ge=1.0, description="Geometric standard deviation of radii (>= 1.0)."
    )
    tol_ov: float = Field(..., gt=0.0, description="Overlap tolerance.")
    n_subcl_percentage: float = Field(
        ..., gt=0.0, le=0.5, description="Target fraction for PCA subcluster size."
    )
    ext_case: int = Field(
        ..., ge=0, le=1, description="CCA sticking ext_case (0 or 1)."
    )
    seed: int | None = Field(None, description="Random seed used for generation.")
    # Add other tunable parameters from config if needed

    model_config = ConfigDict(extra="allow")


class AggregateProperties(BaseModel):
    """Calculated properties of the final generated aggregate."""

    N_particles_actual: int = Field(
        ..., description="Actual number of particles in the final aggregate."
    )
    radius_of_gyration: float | None = Field(
        None, description="Calculated radius of gyration (mass weighted)."
    )
    center_of_mass: list[float] | None = Field(
        None, description="Calculated center of mass [X, Y, Z]."
    )
    # Add r_max etc. if calculated and needed


class GenerationInfo(BaseModel):
    """Information about the generation process."""

    script_name: str = "PyFracVAL"
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Timestamp of generation completion."
    )
    iteration: int = Field(..., description="Aggregate iteration number.")
    # Add git commit hash, hostname, execution time?


class Metadata(BaseModel):
    """Complete output model including parameters, properties, and generation info.

    Designed for easy serialization (e.g., to YAML in header).

    Attributes
    ----------
    generation_info : GenerationInfo
        Information about the run environment and time.
    simulation_parameters : SimulationParameters
        The input parameters used for this simulation run.
    aggregate_properties : AggregateProperties | None
        Calculated properties of the final aggregate (None if calculation failed).
    """

    generation_info: GenerationInfo
    simulation_parameters: SimulationParameters
    aggregate_properties: AggregateProperties | None = (
        None  # Calculated after generation
    )

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            # Add other encoders if needed (e.g., for NumPy types if stored directly)
        }
        # Consider adding validate_assignment = True if you want validation on attribute changes
        # validate_assignment = True
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the metadata model to a dictionary.

        Suitable for YAML/JSON serialization. Uses Pydantic's `model_dump`.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the metadata.
        """
        # mode='json' uses encoders like datetime -> isoformat str
        return self.model_dump(mode="json", exclude_none=True)

    def to_yaml_header(self) -> str:
        """Generate a commented YAML header string for file output.

        Serializes the metadata to a multi-line YAML string where each
        line is prefixed with '# '.

        Returns
        -------
        str
            The formatted YAML header string.
        """
        metadata_dict = self.to_dict()
        # Add comments dynamically if needed for clarity within YAML
        # metadata_dict['simulation_parameters']['N'] = f"{metadata_dict['simulation_parameters']['N']} # Target N" # Example
        yaml_string = yaml.dump(
            metadata_dict,
            sort_keys=False,
            default_flow_style=False,
            indent=2,
            # width=80,
            allow_unicode=True,
        )
        # Prepend comment marker to each line
        header_lines = [f"# {line}\n" for line in yaml_string.splitlines()]
        header_string = "".join(header_lines)
        return header_string

    def save_to_file(
        self, folderpath: str | Path, coords: np.ndarray, radii: np.ndarray
    ):
        """Save metadata (as YAML header) and numerical data to a file.

        Constructs a filename based on simulation parameters and timestamp.
        Writes the YAML header followed by the coordinate and radius data
        formatted as space-delimited columns.

        Parameters
        ----------
        folderpath : str | Path
            The directory where the output file will be saved.
        coords : np.ndarray
            Nx3 NumPy array of final particle coordinates.
        radii : np.ndarray
            N NumPy array of final particle radii.

        Raises
        ------
        IOError
            If writing to the file fails.
        """
        n_str = f"{self.simulation_parameters.N}"
        df_str = f"{self.simulation_parameters.Df:.2f}".replace(".", "p")
        kf_str = f"{self.simulation_parameters.kf:.2f}".replace(".", "p")
        rpg_str = f"{self.simulation_parameters.rp_g:.1f}".replace(".", "p")
        rpgstd_str = f"{self.simulation_parameters.rp_gstd:.2f}".replace(".", "p")
        seed_str = f"{self.simulation_parameters.seed}"  # Use N_A if no seed
        agg_str = f"{self.generation_info.iteration}"
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        filepath = Path(folderpath)
        filepath.mkdir(parents=True, exist_ok=True)
        filepath /= (
            "fracval_"
            + "_".join(
                [
                    f"N{n_str}",
                    f"Df{df_str}",
                    f"kf{kf_str}",
                    f"rpg{rpg_str}",
                    f"rpgstd{rpgstd_str}",
                    # f"seed{seed_str}",
                    f"agg{agg_str}",
                    f"{timestamp}",
                ]
            )
            + ".dat"
        )

        header_string = self.to_yaml_header()
        data_to_save = np.hstack((coords, radii.reshape(-1, 1)))

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header_string)
            np.savetxt(f, data_to_save, fmt="%18.10e", delimiter=" ")
        logger.info("Successfully saved aggregate data and metadata to")
        logger.info(f"    Folder:   {filepath.parent}")
        logger.info(f"    Filename: {filepath.name}")

    @classmethod
    def from_file(cls, filepath: str | Path) -> tuple[Self, np.ndarray]:
        """Load metadata and data from a FracVAL output file.

        Parses the commented YAML header to reconstruct the Metadata object
        and loads the subsequent numerical data into a NumPy array.

        Parameters
        ----------
        filepath : str | Path
            Path to the FracVAL `.dat` file.

        Returns
        -------
        tuple[Metadata | None, np.ndarray | None]
            A tuple containing:
                - The loaded Metadata object, or None if the header is missing,
                  invalid, or fails validation.
                - The loaded Nx4 NumPy data array [X, Y, Z, R], or None if
                  data loading fails or the data is invalid.

        Raises
        ------
        FileNotFoundError
            If the specified `filepath` does not exist.
        Exception
            If YAML parsing fails or loaded data has unexpected dimensions.
        """
        filepath = Path(filepath)
        yaml_lines = []
        data_lines = []

        if not filepath.is_file():
            raise FileNotFoundError(
                f"Metadata load failed: File not found - {filepath}"
            )

        with open(filepath, "r", encoding="utf-8") as f:
            # Read header lines starting with '#'
            for line in f:
                if line.startswith("#"):
                    # Remove the comment marker with whitespace
                    yaml_lines.append(line[2:])
                else:
                    # First non-comment line is data
                    data_lines.append(line)
                    data_lines.extend(f)  # Add rest of file
                    break

        # Try parsing the extracted YAML
        if yaml_lines:
            yaml_string = "".join(yaml_lines)
            metadata_dict = yaml.safe_load(yaml_string)
            if not isinstance(metadata_dict, dict):
                raise Exception(
                    f"Parsed YAML header in {filepath.name} is not a dictionary."
                )
        else:
            logger.warning(f"No commented header lines found in {filepath.name}")

        # Try parsing the numerical data
        data_array = np.loadtxt(filepath)
        if data_array.ndim == 0:
            raise Exception(f"Loaded numerical data is scalar in {filepath.name}.")
        elif data_array.ndim == 1 and data_array.shape[0] == 4:
            data_array = data_array.reshape(1, 4)
        elif data_array.ndim != 2 or data_array.shape[1] != 4:
            logger.warning(
                f"Loaded data array has unexpected shape {data_array.shape} from {filepath.name}. Expected Nx4."
            )

        # Validate and create Metadata model instance *if* metadata was loaded
        metadata_instance = cls(**metadata_dict)
        logger.debug(f"Successfully validated metadata from: {filepath.name}")

        # Return None, None only if file read failed completely at the start
        return metadata_instance, data_array
