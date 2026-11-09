import type { PluginClientContext, PluginSidebarItemProps } from "@getpaseo/plugin/client";
import { SidebarRow } from "@getpaseo/plugin/client/ui";
import { VmmControlScreen } from "./client/greeting";

function VmmSidebarItem({ currentScreen, openScreen }: PluginSidebarItemProps) {
  return (
    <SidebarRow
      icon="Cpu"
      active={currentScreen?.screenId === "vmm-control"}
      onPress={() => openScreen({ screenId: "vmm-control" })}
    />
  );
}

export default function contribute(client: PluginClientContext) {
  client.addScreen({ id: "vmm-control", title: "VMM Swap", Component: VmmControlScreen });
  client.addSidebarHeaderItem({ id: "vmm-control", title: "VMM Swap", Component: VmmSidebarItem });
  return () => {};
}
