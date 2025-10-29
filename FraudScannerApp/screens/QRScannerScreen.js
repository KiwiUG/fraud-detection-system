import React, { useState, useEffect, useRef } from "react";
import { View, Text, TouchableOpacity, ActivityIndicator, StyleSheet } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import axios from "axios";

export default function ScannerScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [scanned, setScanned] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const cameraRef = useRef(null);

  // 🔹 Update this to your computer’s local IP
  const BASE_URL = "http://10.34.67.151:8000";

  // --- Handle camera permission ---
  if (!permission) return <View />;
  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>Camera access is required to scan QR codes.</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

const handleBarCodeScanned = async ({ data }) => {
  if (scanned) return;
  setScanned(true);
  setLoading(true);
  setResult(null);

  try {
    // Remove the prefix if QR code contains "USER_ID|"
    const userId = data.includes("USER_ID|") ? data.split("|")[1] : data;

    const response = await axios.get(`${BASE_URL}/reputation/${userId}`);
    setResult(response.data);
  } catch (error) {
    console.error(error);
    setResult({ error: "❌ Network Error: Could not reach backend." });
  } finally {
    setLoading(false);
  }
};


  const getRiskColor = (risk) => {
    switch (risk) {
      case "HIGH": return "#ff4d4d";
      case "MEDIUM": return "#ffa31a";
      default: return "#4CAF50";
    }
  };

  return (
    <View style={styles.container}>
      {!scanned ? (
<View style={{ flex: 1 }}>
  <CameraView
    ref={cameraRef}
    style={StyleSheet.absoluteFill}
    facing="back"
    barcodeScannerSettings={{ barcodeTypes: ["qr"] }}
    onBarcodeScanned={handleBarCodeScanned}
  />
  <View style={styles.overlay}>
    <Text style={styles.overlayText}>Align QR Code inside the frame</Text>
  </View>
</View>

      ) : (
        <View style={styles.resultContainer}>
          {loading ? (
            <ActivityIndicator size="large" color="#007AFF" />
          ) : result ? (
            result.error ? (
              <View style={[styles.card, { borderColor: "#ff4d4d" }]}>
                <Text style={[styles.title, { color: "#ff4d4d" }]}>{result.error}</Text>
              </View>
            ) : (
              <View style={[styles.card, { borderColor: getRiskColor(result.risk_level) }]}>
                <Text style={[styles.title, { color: getRiskColor(result.risk_level) }]}>
                  {result.risk_level} RISK
                </Text>
                <Text style={styles.percentage}>{result.risk_percentage}% Risk</Text>
                <Text style={styles.detail}>{result.message}</Text>
                <Text style={styles.transactions}>
                  Transactions Analyzed: {result.transactions_analyzed}
                </Text>
              </View>
            )
          ) : null}

          <TouchableOpacity
            style={styles.button}
            onPress={() => {
              setScanned(false);
              setResult(null);
            }}
          >
            <Text style={styles.buttonText}>🔄 Scan Again</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  camera: {
    flex: 1,
    justifyContent: "flex-end",
  },
  overlay: {
    backgroundColor: "rgba(0, 0, 0, 0.4)",
    alignItems: "center",
    padding: 16,
  },
  overlayText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "500",
  },
  resultContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#121212",
    padding: 20,
  },
  card: {
    width: "100%",
    borderWidth: 2,
    borderRadius: 16,
    padding: 20,
    backgroundColor: "#1e1e1e",
    marginBottom: 20,
    alignItems: "center",
  },
  title: {
    fontSize: 26,
    fontWeight: "bold",
    marginBottom: 8,
  },
  percentage: {
    fontSize: 22,
    color: "#fff",
    marginBottom: 8,
  },
  detail: {
    fontSize: 16,
    color: "#ccc",
    textAlign: "center",
  },
  transactions: {
    marginTop: 8,
    fontSize: 14,
    color: "#999",
  },
  button: {
    backgroundColor: "#007AFF",
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  permissionContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  permissionText: {
    textAlign: "center",
    color: "#fff",
    marginBottom: 10,
  },
});
