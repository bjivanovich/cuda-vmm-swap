import type { PluginServerContext } from "@getpaseo/plugin/server";
import { getVmmStatus, swapVmmModel } from "./server/greeting";
import { vmmStatusRpc, vmmSwapRpc } from "./shared/greeting";

export default function contribute(server: PluginServerContext) {
  server.handle(vmmStatusRpc, getVmmStatus);
  server.handle(vmmSwapRpc, swapVmmModel);
  return () => {};
}
