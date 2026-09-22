import { app } from "../../../scripts/app.js";

const NODE_NAME = "SelfLiftH3Sampler";

function setWidgetVisible(node, widget, visible) {
    if (!widget) {
        return;
    }

    const hidden = !visible;
    widget.options ??= {};
    if (widget.hidden === hidden && widget.options.hidden === hidden) {
        return;
    }

    widget.hidden = hidden;
    widget.options.hidden = hidden;

    const computedSize = node.computeSize();
    node.setSize([Math.max(node.size[0], computedSize[0]), computedSize[1]]);
    node.setDirtyCanvas(true, true);
}

function syncTilingWidgetVisibility(node) {
    const tilingWidget = node.widgets?.find((widget) => widget.name === "highres_tiling");
    const tilingEnabled = tilingWidget?.value === true;
    for (const name of ["highres_tile_count", "highres_tile_axis"]) {
        setWidgetVisible(node, node.widgets?.find((widget) => widget.name === name), tilingEnabled);
    }
}

app.registerExtension({
    name: "SelfLift.DynamicWidgets",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            const tilingWidget = this.widgets?.find((widget) => widget.name === "highres_tiling");

            if (tilingWidget) {
                const node = this;
                const originalCallback = tilingWidget.callback;
                tilingWidget.callback = function () {
                    const callbackResult = originalCallback?.apply(this, arguments);
                    syncTilingWidgetVisibility(node);
                    return callbackResult;
                };
            }

            syncTilingWidgetVisibility(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            syncTilingWidgetVisibility(this);
            return result;
        };
    },
});
