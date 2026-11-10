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
      badgeText: {
        fontSize: 10,
        fontWeight: "bold" as const,
        color: theme.colors.foreground,
      },
      badgeTextActive: {
        color: theme.colors.accentForeground,
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
        opacity: 0.5,
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
      secondaryButtonText: {
        fontSize: 11,
        color: theme.colors.foreground,
      },
      dangerButton: {
        paddingVertical: 8,
        paddingHorizontal: 12,
        backgroundColor: theme.colors.statusDanger ?? "#e06c75",
        borderRadius: 6,
        alignItems: "center" as const,
        justifyContent: "center" as const,
      },
      dangerButtonText: {
        fontSize: 11,
        fontWeight: "bold" as const,
        color: "#ffffff",
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
    }),
    [theme, layout.compact],
  );

  const status = statusQuery.data;
  const models = status?.models ?? [];

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.scrollContent}>
      <Text style={styles.header}>[CUDA-VMM-SWAP] Studio</Text>
      <Text style={styles.subHeader}>
        Dynamic N-Agent VMM Hot-Swap Engine | Windows WDDM Aware
      </Text>

      {/* TELEMETRY CARD */}
      <View style={styles.card}>
        <View style={styles.statRow}>
          <Text style={styles.label}>Active in VRAM:</Text>
          <View style={[styles.badge, status?.active_vram_model ? styles.badgeActive : null]}>
            <Text style={[styles.badgeText, status?.active_vram_model ? styles.badgeTextActive : null]}>
              {(status?.active_vram_model ?? "OFFLINE").toUpperCase()}
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
          <Text style={styles.value}>{status?.swap_enabled ? "DYNAMIC SWAP" : "SINGLE MODEL"}</Text>
        </View>

        {/* ACTION CONTROLS */}
        <View style={[styles.buttonRow, { marginTop: 6 }]}>
          <Pressable
            accessibilityRole="button"
            style={[styles.secondaryButton, { flex: 1 }]}
            onPress={() => toggleMutation.mutate({})}
          >
            <Text style={styles.secondaryButtonText}>
              {status?.swap_enabled ? "Disable Swap" : "Enable Swap"}
            </Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            style={[styles.dangerButton, { flex: 1 }]}
            onPress={() => evictMutation.mutate({})}
          >
            <Text style={styles.dangerButtonText}>Evict VRAM (&lt;60ms)</Text>
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
              style={[styles.button, isCurrent ? styles.buttonDisabled : null, { marginTop: 4 }]}
              disabled={isCurrent}
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
