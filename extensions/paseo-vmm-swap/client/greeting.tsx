import type { PluginScreenProps } from "@getpaseo/plugin/client";
import { useRpc } from "@getpaseo/plugin/client";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { Pressable, Text, View } from "react-native";
import { vmmStatusRpc, vmmSwapRpc } from "../shared/greeting";
import { openExternal } from "./web";

export function VmmControlScreen({ theme, layout }: PluginScreenProps) {
  const getStatus = useRpc(vmmStatusRpc);
  const swapModel = useRpc(vmmSwapRpc);

  const statusQuery = useQuery({
    queryKey: ["vmm", "status"],
    queryFn: () => getStatus({}),
    refetchInterval: 3000,
  });

  const swapMutation = useMutation({
    mutationFn: swapModel,
    onSuccess: () => {
      statusQuery.refetch();
    },
  });

  const styles = useMemo(
    () => ({
      screen: {
        flex: 1,
        padding: layout.compact ? 16 : 24,
        backgroundColor: theme.colors.surface0,
      },
      header: {
        fontSize: 16,
        fontWeight: "bold" as const,
        color: theme.colors.foreground,
        marginBottom: 8,
      },
      subHeader: {
        fontSize: 12,
        color: theme.colors.mutedForeground,
        marginBottom: 16,
      },
      card: {
        backgroundColor: theme.colors.surface1,
        padding: 14,
        borderRadius: 8,
        marginBottom: 16,
        borderWidth: 1,
        borderColor: theme.colors.border,
      },
      statRow: {
        flexDirection: "row" as const,
        justifyContent: "space-between" as const,
        marginBottom: 6,
      },
      label: {
        fontSize: 12,
        color: theme.colors.mutedForeground,
      },
      value: {
        fontSize: 12,
        fontWeight: "bold" as const,
        color: theme.colors.foreground,
      },
      buttonRow: {
        flexDirection: "row" as const,
        gap: 8,
        marginTop: 10,
        marginBottom: 10,
      },
      button: {
        flex: 1,
        padding: 10,
        backgroundColor: theme.colors.accent,
        borderRadius: 6,
        alignItems: "center" as const,
      },
      buttonText: {
        fontSize: 12,
        fontWeight: "bold" as const,
        color: theme.colors.accentForeground,
      },
      secondaryButton: {
        padding: 10,
        backgroundColor: theme.colors.surface2,
        borderRadius: 6,
        alignItems: "center" as const,
        marginTop: 8,
      },
      secondaryButtonText: {
        fontSize: 12,
        color: theme.colors.foreground,
      },
    }),
    [theme, layout.compact],
  );

  const status = statusQuery.data;

  return (
    <View style={styles.screen}>
      <Text style={styles.header}>[CUDA-VMM-SWAP] Studio</Text>
      <Text style={styles.subHeader}>Windows WDDM Pipelined VMM Hot-Swap Engine</Text>

      <View style={styles.card}>
        <View style={styles.statRow}>
          <Text style={styles.label}>Active Model in VRAM:</Text>
          <Text style={styles.value}>{(status?.active_vram_model ?? "OFFLINE").toUpperCase()}</Text>
        </View>
        <View style={styles.statRow}>
          <Text style={styles.label}>Physical VRAM Used:</Text>
          <Text style={styles.value}>{status?.physical_vram_used_mb ?? 0} MB</Text>
        </View>
        <View style={styles.statRow}>
          <Text style={styles.label}>Last Swap Latency:</Text>
          <Text style={styles.value}>{(status?.last_swap_latency_seconds ?? 0).toFixed(3)} s</Text>
        </View>
        <View style={styles.statRow}>
          <Text style={styles.label}>Swap Mode:</Text>
          <Text style={styles.value}>{status?.swap_enabled ? "ENABLED" : "DISABLED"}</Text>
        </View>
      </View>

      <Text style={[styles.label, { marginBottom: 6 }]}>Manual VMM Hot-Swap (&lt;2.3s DMA):</Text>
      <View style={styles.buttonRow}>
        <Pressable
          accessibilityRole="button"
          style={styles.button}
          onPress={() => swapMutation.mutate({ model: "architect" })}
        >
          <Text style={styles.buttonText}>Architect</Text>
        </Pressable>
        <Pressable
          accessibilityRole="button"
          style={styles.button}
          onPress={() => swapMutation.mutate({ model: "coder" })}
        >
          <Text style={styles.buttonText}>Coder</Text>
        </Pressable>
        <Pressable
          accessibilityRole="button"
          style={styles.button}
          onPress={() => swapMutation.mutate({ model: "debugger" })}
        >
          <Text style={styles.buttonText}>Debugger</Text>
        </Pressable>
      </View>

      <Pressable
        accessibilityRole="link"
        style={styles.secondaryButton}
        onPress={() => openExternal("http://127.0.0.1:8000/ui")}
      >
        <Text style={styles.secondaryButtonText}>Open Web Control Studio in Browser</Text>
      </Pressable>
    </View>
  );
}
