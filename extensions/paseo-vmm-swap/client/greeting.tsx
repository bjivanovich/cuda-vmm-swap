import type { PluginScreenProps } from "@getpaseo/plugin/client";
import { useRpc } from "@getpaseo/plugin/client";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Pressable, ScrollView, Text, TextInput, View } from "react-native";
import {
  vmmConfigureModelRpc,
  vmmEvictRpc,
  vmmStatusRpc,
  vmmSwapRpc,
  vmmToggleSwapRpc,
} from "../shared/greeting";
import { openExternal } from "./web";

export function VmmControlScreen({ theme, layout }: PluginScreenProps) {
  const getStatus = useRpc(vmmStatusRpc);
  const swapModel = useRpc(vmmSwapRpc);
  const toggleSwap = useRpc(vmmToggleSwapRpc);
  const evictAll = useRpc(vmmEvictRpc);
  const configureModel = useRpc(vmmConfigureModelRpc);

  const [formOpen, setFormOpen] = useState(false);
  const [roleId, setRoleId] = useState("");
  const [modelPath, setModelPath] = useState("");
  const [alias, setAlias] = useState("");
  const [roleDesc, setRoleDesc] = useState("");
  const [contextTokens, setContextTokens] = useState("98304");
  const [plePath, setPlePath] = useState("");
  const [formMsg, setFormMsg] = useState("");

  const statusQuery = useQuery({
    queryKey: ["vmm", "status"],
    queryFn: () => getStatus({}),
    refetchInterval: 3000,
  });

  const swapMutation = useMutation({
    mutationFn: swapModel,
    onSuccess: () => statusQuery.refetch(),
  });

  const toggleMutation = useMutation({
    mutationFn: toggleSwap,
    onSuccess: () => statusQuery.refetch(),
  });

  const evictMutation = useMutation({
    mutationFn: evictAll,
    onSuccess: () => statusQuery.refetch(),
  });

  const configureMutation = useMutation({
    mutationFn: configureModel,
    onSuccess: (data) => {
      setFormMsg(`Saved slot: ${data.model_id}`);
      setRoleId("");
      setModelPath("");
      setAlias("");
      setRoleDesc("");
      setPlePath("");
      statusQuery.refetch();
    },
    onError: (err: any) => {
      setFormMsg(`Error: ${err.message}`);
    },
  });

  const styles = useMemo(
    () => ({
      screen: {
        flex: 1,
        backgroundColor: theme.colors.surface0,
      },
      scrollContent: {
        padding: layout.compact ? 14 : 20,
        gap: 12,
      },
      headerRow: {
        flexDirection: "row" as const,
        justifyContent: "space-between" as const,
        alignItems: "center" as const,
      },
      header: {
        fontSize: 16,
        fontWeight: "bold" as const,
        color: theme.colors.foreground,
      },
      subHeader: {
        fontSize: 12,
        color: theme.colors.foregroundMuted,
        marginBottom: 4,
      },
      card: {
        backgroundColor: theme.colors.surface1,
        padding: 14,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: theme.colors.border,
        gap: 6,
      },
      cardActive: {
        borderColor: theme.colors.accent,
        borderWidth: 2,
      },
      sectionTitle: {
        fontSize: 13,
        fontWeight: "bold" as const,
        color: theme.colors.foreground,
        marginTop: 6,
      },
      statRow: {
        flexDirection: "row" as const,
        justifyContent: "space-between" as const,
        alignItems: "center" as const,
      },
      label: {
        fontSize: 12,
        color: theme.colors.foregroundMuted,
      },
      value: {
        fontSize: 12,
        fontWeight: "bold" as const,
        color: theme.colors.foreground,
      },
      badge: {
        paddingHorizontal: 8,
        paddingVertical: 2,
        borderRadius: 4,
        backgroundColor: theme.colors.surface2,
      },
      badgeActive: {
        backgroundColor: theme.colors.accent,
      },
      badgeOnline: {
        backgroundColor: theme.colors.statusSuccess ?? "#3fb950",
      },
      badgeOffline: {
        backgroundColor: theme.colors.statusDanger ?? "#e06c75",
      },
      badgeText: {
        fontSize: 10,
        fontWeight: "bold" as const,
        color: theme.colors.foreground,
      },
      badgeTextActive: {
        color: theme.colors.accentForeground,
      },
      badgeTextStatus: {
        color: "#ffffff",
      },
      buttonRow: {
        flexDirection: "row" as const,
        gap: 8,
      },
      button: {
        paddingVertical: 8,
        paddingHorizontal: 12,
        backgroundColor: theme.colors.accent,
        borderRadius: 6,
        alignItems: "center" as const,
        justifyContent: "center" as const,
      },
      buttonDisabled: {
        opacity: 0.4,
      },
      buttonText: {
        fontSize: 11,
        fontWeight: "bold" as const,
        color: theme.colors.accentForeground,
      },
      secondaryButton: {
        paddingVertical: 8,
        paddingHorizontal: 12,
        backgroundColor: theme.colors.surface2,
        borderRadius: 6,
        alignItems: "center" as const,
        justifyContent: "center" as const,
      },
      secondaryButtonActive: {
        backgroundColor: theme.colors.accent,
      },
      secondaryButtonText: {
        fontSize: 11,
        color: theme.colors.foreground,
      },
      secondaryButtonTextActive: {
        fontSize: 11,
        fontWeight: "bold" as const,
        color: theme.colors.accentForeground,
      },
      dangerButton: {
        paddingVertical: 8,
        paddingHorizontal: 12,
        backgroundColor: theme.colors.statusDanger ?? "#e06c75",
        borderRadius: 6,
        alignItems: "center" as const,
        justifyContent: "center" as const,
      },
      dangerButtonDisabled: {
        backgroundColor: theme.colors.surface2,
        opacity: 0.5,
      },
      dangerButtonText: {
        fontSize: 11,
        fontWeight: "bold" as const,
        color: "#ffffff",
      },
      dangerButtonTextDisabled: {
        fontSize: 11,
        color: theme.colors.foregroundMuted,
      },
      input: {
        backgroundColor: theme.colors.surface2,
        color: theme.colors.foreground,
        padding: 8,
        borderRadius: 6,
        fontSize: 12,
        borderWidth: 1,
        borderColor: theme.colors.border,
      },
      inputLabel: {
        fontSize: 11,
        color: theme.colors.foregroundMuted,
        marginTop: 4,
      },
      bannerOffline: {
        backgroundColor: theme.colors.surface1,
        borderLeftWidth: 4,
        borderLeftColor: theme.colors.statusDanger ?? "#e06c75",
        padding: 10,
        borderRadius: 6,
        gap: 4,
      },
      bannerOfflineTitle: {
        fontSize: 12,
        fontWeight: "bold" as const,
        color: theme.colors.statusDanger ?? "#e06c75",
      },
      bannerOfflineText: {
        fontSize: 11,
        color: theme.colors.foregroundMuted,
      },
    }),
    [theme, layout.compact],
  );

  const status = statusQuery.data;
  const isConnected = Boolean(status?.connected);
  const activeModel = status?.active_vram_model;
  const hasActiveModel = Boolean(activeModel && activeModel !== "offline");
  const models = status?.models ?? [];
  const swapEnabled = Boolean(status?.swap_enabled);

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.scrollContent}>
      <View style={styles.headerRow}>
        <Text style={styles.header}>[CUDA-VMM-SWAP] Studio</Text>
        <View style={[styles.badge, isConnected ? styles.badgeOnline : styles.badgeOffline]}>
          <Text style={[styles.badgeText, styles.badgeTextStatus]}>
            {isConnected ? "ROUTER ONLINE (:8000)" : "ROUTER OFFLINE"}
          </Text>
        </View>
      </View>
      <Text style={styles.subHeader}>
        Dynamic N-Agent VMM Hot-Swap Engine | Windows WDDM Aware
      </Text>

      {/* OFFLINE WARNING BANNER */}
      {!isConnected && (
        <View style={styles.bannerOffline}>
          <Text style={styles.bannerOfflineTitle}>VMM Router Not Responding on 127.0.0.1:8000</Text>
          <Text style={styles.bannerOfflineText}>
            Launch the backend supervisor in your terminal:
          </Text>
          <Text style={[styles.value, { fontSize: 11, color: theme.colors.foreground }]}>
            python master_harness.py
          </Text>
          <Text style={[styles.bannerOfflineText, { fontSize: 10 }]}>
            (Or: python router/vmm_router.py)
          </Text>
        </View>
      )}

      {/* TELEMETRY CARD */}
      <View style={styles.card}>
        <View style={styles.statRow}>
          <Text style={styles.label}>Active in VRAM:</Text>
          <View style={[styles.badge, hasActiveModel ? styles.badgeActive : null]}>
            <Text style={[styles.badgeText, hasActiveModel ? styles.badgeTextActive : null]}>
              {hasActiveModel ? activeModel!.toUpperCase() : (isConnected ? "NONE (FREE VRAM)" : "OFFLINE")}
            </Text>
          </View>
        </View>
        <View style={styles.statRow}>
          <Text style={styles.label}>Physical VRAM Used:</Text>
          <Text style={styles.value}>{status?.physical_vram_used_mb ?? 0} MB</Text>
        </View>
        <View style={styles.statRow}>
          <Text style={styles.label}>Last Swap Latency:</Text>
          <Text style={styles.value}>
            {(status?.last_swap_latency_seconds ?? 0).toFixed(3)} s
          </Text>
        </View>
        <View style={styles.statRow}>
          <Text style={styles.label}>Swap Mode:</Text>
          <View style={[styles.badge, swapEnabled ? styles.badgeActive : null]}>
            <Text style={[styles.badgeText, swapEnabled ? styles.badgeTextActive : null]}>
              {swapEnabled ? "DYNAMIC N-SWAP: ON" : "SINGLE-MODEL: OFF"}
            </Text>
          </View>
        </View>

        {/* ACTION CONTROLS */}
        <View style={[styles.buttonRow, { marginTop: 6 }]}>
          <Pressable
            accessibilityRole="button"
            style={[
              styles.secondaryButton,
              swapEnabled ? styles.secondaryButtonActive : null,
              !isConnected ? styles.buttonDisabled : null,
              { flex: 1 },
            ]}
            disabled={!isConnected}
            onPress={() => toggleMutation.mutate({})}
          >
            <Text style={swapEnabled ? styles.secondaryButtonTextActive : styles.secondaryButtonText}>
              {swapEnabled ? "Disable Swap Mode" : "Enable Swap Mode"}
            </Text>
          </Pressable>

          <Pressable
            accessibilityRole="button"
            style={[
              hasActiveModel ? styles.dangerButton : styles.dangerButtonDisabled,
              !isConnected || !hasActiveModel ? styles.buttonDisabled : null,
              { flex: 1 },
            ]}
            disabled={!isConnected || !hasActiveModel}
            onPress={() => evictMutation.mutate({})}
          >
            <Text style={hasActiveModel ? styles.dangerButtonText : styles.dangerButtonTextDisabled}>
              {hasActiveModel ? `Evict VRAM (<60ms)` : `VRAM Clear`}
            </Text>
          </Pressable>
        </View>
      </View>

      {/* CONFIGURED N-AGENT SLOTS */}
      <Text style={styles.sectionTitle}>
        Configured Agent Models ({models.length} Slots):
      </Text>

      {models.map((m) => {
        const isCurrent = m.is_active;
        return (
          <View key={m.id} style={[styles.card, isCurrent ? styles.cardActive : null]}>
            <View style={styles.statRow}>
              <View>
                <Text style={[styles.value, { fontSize: 13 }]}>{m.alias} ({m.id})</Text>
                <Text style={[styles.label, { fontSize: 10 }]}>{m.role}</Text>
              </View>
              <View style={[styles.badge, isCurrent ? styles.badgeActive : null]}>
                <Text style={[styles.badgeText, isCurrent ? styles.badgeTextActive : null]}>
                  {isCurrent ? "ACTIVE IN VRAM" : "STANDBY"}
                </Text>
              </View>
            </View>

            <Text style={[styles.label, { fontSize: 10, marginTop: 4 }]} numberOfLines={1}>
              {m.model_path || "Path not set"}
            </Text>

            <View style={styles.statRow}>
              <Text style={styles.label}>Context: {m.context.toLocaleString()} tok</Text>
              <Text style={styles.label}>KV: {m.ctk}/{m.ctv}</Text>
            </View>

            <Pressable
              accessibilityRole="button"
              style={[
                styles.button,
                isCurrent || !isConnected ? styles.buttonDisabled : null,
                { marginTop: 4 },
              ]}
              disabled={isCurrent || !isConnected}
              onPress={() => swapMutation.mutate({ model: m.id })}
            >
              <Text style={styles.buttonText}>
                {isCurrent ? "Active in VRAM" : `Swap to VRAM (<2.3s)`}
              </Text>
            </Pressable>
          </View>
        );
      })}

      {/* DYNAMIC AGENT CONFIGURATION / ADD AGENT */}
      <Pressable
        accessibilityRole="button"
        style={[styles.secondaryButton, { marginTop: 4 }]}
        onPress={() => setFormOpen((prev) => !prev)}
      >
        <Text style={styles.secondaryButtonText}>
          {formOpen ? "▼ Close Agent Configuration" : "+ Add / Choose Custom Agent Model"}
        </Text>
      </Pressable>

      {formOpen && (
        <View style={styles.card}>
          <Text style={[styles.value, { marginBottom: 4 }]}>Configure / Add Agent Slot</Text>

          <Text style={styles.inputLabel}>Agent Role ID (e.g. coder, architect, tester, reviewer):</Text>
          <TextInput
            style={styles.input}
            value={roleId}
            onChangeText={setRoleId}
            placeholder="e.g. coder, tester, analyst"
            placeholderTextColor={theme.colors.foregroundMuted}
            autoCapitalize="none"
          />

          <Text style={styles.inputLabel}>GGUF Model Path:</Text>
          <TextInput
            style={styles.input}
            value={modelPath}
            onChangeText={setModelPath}
            placeholder="D:/models/your-model.gguf"
            placeholderTextColor={theme.colors.foregroundMuted}
            autoCapitalize="none"
          />

          <Text style={styles.inputLabel}>Friendly Alias:</Text>
          <TextInput
            style={styles.input}
            value={alias}
            onChangeText={setAlias}
            placeholder="e.g. Qwen3.8-Coder-TQ2"
            placeholderTextColor={theme.colors.foregroundMuted}
          />

          <Text style={styles.inputLabel}>Role Description:</Text>
          <TextInput
            style={styles.input}
            value={roleDesc}
            onChangeText={setRoleDesc}
            placeholder="e.g. Specialized Unit Test Engineer"
            placeholderTextColor={theme.colors.foregroundMuted}
          />

          <Text style={styles.inputLabel}>Context Window (Tokens):</Text>
          <TextInput
            style={styles.input}
            value={contextTokens}
            onChangeText={setContextTokens}
            keyboardType="numeric"
            placeholder="98304"
            placeholderTextColor={theme.colors.foregroundMuted}
          />

          <Text style={styles.inputLabel}>Modular PLE Table (Optional):</Text>
          <TextInput
            style={styles.input}
            value={plePath}
            onChangeText={setPlePath}
            placeholder="D:/models/ple-table.gguf"
            placeholderTextColor={theme.colors.foregroundMuted}
            autoCapitalize="none"
          />

          {formMsg ? (
            <Text style={[styles.label, { color: theme.colors.accent, marginTop: 4 }]}>
              {formMsg}
            </Text>
          ) : null}

          <Pressable
            accessibilityRole="button"
            style={[styles.button, { marginTop: 8 }]}
            onPress={() => {
              if (!roleId.trim() || !modelPath.trim()) {
                setFormMsg("Error: Role ID and Model Path are required.");
                return;
              }
              configureMutation.mutate({
                id: roleId.trim(),
                model_path: modelPath.trim(),
                alias: alias.trim() || undefined,
                role: roleDesc.trim() || undefined,
                context: parseInt(contextTokens, 10) || 98304,
                ple_path: plePath.trim() || undefined,
              });
            }}
          >
            <Text style={styles.buttonText}>Save Agent Slot to Router</Text>
          </Pressable>
        </View>
      )}

      {/* EXTERNAL STUDIO LINK */}
      <Pressable
        accessibilityRole="link"
        style={[styles.secondaryButton, { marginTop: 6 }]}
        onPress={() => openExternal("http://127.0.0.1:8000/ui")}
      >
        <Text style={styles.secondaryButtonText}>Open Web Control Studio in Browser</Text>
      </Pressable>
    </ScrollView>
  );
}
