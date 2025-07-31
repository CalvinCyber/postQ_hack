import qiskit
import json
import os
import numpy as np
from ecdsa import SigningKey, SECP256k1
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_algorithms import AmplificationProblem, Grover
from keras.models import Sequential
from keras.layers import Dense
from qiskit.circuit.library import MCXGate

# ✅ Generate the next filename dynamically
def get_next_filename(base_filename='data', extension='json'):
    i = 0
    while True:
        filename = f"{base_filename}_{i}.{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1

def save_data_to_json(new_data, base_filename='metadata/data_hacked', extension='json'):
    filename = get_next_filename(base_filename, extension)
    with open(filename, 'w') as json_file:
        json.dump(new_data, json_file, indent=4)
    print(f"Data saved to {filename}")

# ✅ Generate ECDSA Key Pair
ecdsa_private_key = SigningKey.generate(curve=SECP256k1)
ecdsa_public_key = ecdsa_private_key.verifying_key.to_string().hex()

# ✅ Convert Public Key to Binary for SECRET_KEY
SECRET_KEY = bin(int(ecdsa_public_key, 16))[2:].zfill(256)  # 256-bit binary key
print(f"Derived Secret Key (first 32 bits): {SECRET_KEY[:20]}...")
MAX_QUBITS = min(20, len(SECRET_KEY))  # Use only 20 qubits max
SECRET_KEY = SECRET_KEY[:MAX_QUBITS]  # Truncate to fit

print(f"Using {MAX_QUBITS} qubits for Grover's algorithm.")
print(SECRET_KEY)
# ✅ IBM Quantum Setup
service = QiskitRuntimeService(channel="ibm_quantum", token="###")
available_backends = service.backends()
print("Available backends:", available_backends)
backend = service.backend("ibm_kyiv")
sampler = Sampler(backend)

# ✅ Define Grover's Oracle (Marks SECRET_KEY as solution)
n = len(SECRET_KEY)  # Number of qubits
oracle = QuantumCircuit(n)

# Apply X gates to match the secret key (where bit is '0')
for qubit, bit in enumerate(SECRET_KEY):
    if bit == "0":
        oracle.x(qubit)

oracle.h(n - 1)  # Hadamard on the target qubit

# Use MCXGate instead of the deprecated mct
mcx_gate = MCXGate(n - 1)
oracle.append(mcx_gate, list(range(n - 1)) + [n - 1])

oracle.h(n - 1)  # Hadamard again

# Apply X gates back
for qubit, bit in enumerate(SECRET_KEY):
    if bit == "0":
        oracle.x(qubit)

# ✅ **Transpile the Oracle for IBM Quantum Backend**
oracle = transpile(oracle, backend)

# ✅ Define Grover’s Algorithm
problem = AmplificationProblem(oracle)
grover = Grover(iterations=1, sampler=sampler)  # Set a fixed iteration count
grover_circuit = grover.construct_circuit(problem)  # Now it works
print('transpiling')
grover_circuit = transpile(grover_circuit, backend)  # Transpile for backend
print(grover_circuit)
grover._iterations = list(grover._iterations)  # Convert generator to list
print(grover._iterations)
# grover_circuit = grover.construct_circuit(problem)
# grover_circuit = transpile(grover_circuit, backend)  # Transpile for compatibility

# ✅ Run the quantum search
job = sampler.run([grover_circuit])
result = job.result()
print(result)
# ✅ **Transpile the full Grover circuit before running**
# result = grover.amplify(problem)  # ✅ Directly amplify instead of constructing the circuit

# ✅ Extract the quantum result safely
if result.circuit_results:
    quantum_secret_key = max(result.circuit_results, key=result.circuit_results.get)
    print(f"Quantum Computed Secret Key: {quantum_secret_key}")
else:
    print("Grover's algorithm failed to return a valid result.")
    quantum_secret_key = None  # Handle failure case

# ✅ Get the quantum result
# quantum_secret_key = max(result.circuit_results, key=result.circuit_results.get)
# print(f"Quantum Computed Secret Key: {quantum_secret_key}")

# ✅ AI Model Prediction of Private Keys
def generate_ecdsa_data(samples=10000):
    private_keys = []
    public_keys = []
    for _ in range(samples):
        sk = SigningKey.generate(curve=SECP256k1)
        pk = sk.verifying_key.to_string().hex()
        private_keys.append(int(sk.to_string().hex(), 16))
        public_keys.append(int(pk, 16))
    return np.array(private_keys), np.array(public_keys)

X_train, y_train = generate_ecdsa_data(samples=10000)

model = Sequential([
    Dense(256, activation='relu', input_shape=(1,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

test_public_key = np.array([int(ecdsa_public_key, 16)])
predicted_private_key = model.predict(test_public_key)

print(f"Predicted Private Key: {hex(int(predicted_private_key[0]))}")

# ✅ Save results
data = {
    'secret_key': SECRET_KEY,
    'quantum_computed_key': quantum_secret_key,
    'address': ecdsa_public_key,
    'true_key': ecdsa_private_key.to_string().hex(),
    'AI_Key': hex(int(predicted_private_key[0]))
}
save_data_to_json(data)
