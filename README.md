# 🔐 Quantum & AI-Based ECDSA Key Recovery

This project demonstrates an experimental approach to combining **Quantum Computing** (via Grover's Algorithm) with **Artificial Intelligence (Neural Networks)** to predict and recover ECDSA (Elliptic Curve Digital Signature Algorithm) keys.

---

## 📦 Features

- ✅ Quantum search using **Grover’s Algorithm** on IBM Quantum backends  
- ✅ Oracle circuit construction for a secret ECDSA-derived key  
- ✅ Neural network trained to predict ECDSA private keys from public keys  
- ✅ Automatic saving of all results to versioned JSON files  
- ✅ Uses real elliptic curve crypto and quantum sampling via `qiskit-ibm-runtime`

---

## 🧠 Concepts Used

- **Grover's Algorithm**: Quantum algorithm to find a target input with quadratic speed-up  
- **ECDSA Key Pair Generation**: Elliptic curve cryptography using `SECP256k1`  
- **Neural Networks**: Built with Keras to regress private keys from public keys  
- **Quantum Circuit Oracle**: Dynamically constructed based on the binary representation of public key

---
