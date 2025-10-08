import sys
sys.path.append("../../")

import pickle
from datetime import datetime, timezone

from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit_ibm_runtime.models.backend_properties import BackendProperties as IBMBackendProperties

from quantum_molecular_encodings.paths import BACKENDS_DIR


def save_noise_model(noise_model, path):
    """Save a noise model to disk with timestamp metadata."""
    setattr(noise_model, 'datetime_saved', datetime.now())
    with open(path, "wb") as f:
        pickle.dump(noise_model, f)


def save_backend_target(target, path):
    """Save a backend target to disk."""
    with open(path, "wb") as f:
        pickle.dump(target, f)


# Configuration
year = 2025
month = 8
day = 10
hour = 18
minute = 59
second = 40
target_time = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)

device = "ibm_pittsburgh"

# Initialize Qiskit Runtime Service
service = QiskitRuntimeService()
backend = service.backend(device)

# Get backend properties at target time
props_target = backend.properties(datetime=target_time)
props_target_dict = props_target.to_dict()
props_target_converted = BackendProperties.from_dict(
    IBMBackendProperties.from_dict(props_target_dict).to_dict()
)

# Create noise model from backend properties
noise_model = NoiseModel.from_backend_properties(props_target_converted)

# Generate filename timestamp
timestamp_str = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}+00:00"
base_filename = f"{device}_{timestamp_str}"

# Save noise model
save_noise_model(
    noise_model,
    BACKENDS_DIR / f"{base_filename}_noise_model.pkl"
)

# Get and save backend target history
target_history = backend.target_history(datetime=target_time)
save_backend_target(
    target_history,
    BACKENDS_DIR / f"{base_filename}_target.pkl"
)