import type { PluginServerContext } from "@getpaseo/plugin/server";
import {
  configureVmmModel,
  evictVmmAll,
  getVmmStatus,
  swapVmmModel,
  toggleVmmSwap,
} from "./server/greeting";
import {
  vmmConfigureModelRpc,
  vmmEvictRpc,
  vmmStatusRpc,
  vmmSwapRpc,
  vmmToggleSwapRpc,
} from "./shared/greeting";

export default function contribute(server: PluginServerContext) {
  server.handle(vmmStatusRpc, getVmmStatus);
  server.handle(vmmSwapRpc, swapVmmModel);
  server.handle(vmmToggleSwapRpc, toggleVmmSwap);
  server.handle(vmmEvictRpc, evictVmmAll);
  server.handle(vmmConfigureModelRpc, configureVmmModel);
  return () => {};
}
