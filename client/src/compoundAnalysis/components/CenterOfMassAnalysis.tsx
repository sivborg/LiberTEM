import * as React from "react";
import { useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Checkbox, Form, Header, Icon, Input, Modal, Popup } from "semantic-ui-react";
import { defaultDebounce } from "../../helpers";
import ResultList from "../../job/components/ResultList";
import { AnalysisTypes } from "../../messages";
import { RootReducer } from "../../store";
import { cbToRadius, inRectConstraint, keepOnCY } from "../../widgets/constraints";
import Disk from "../../widgets/Disk";
import { DraggableHandle } from "../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../widgets/types";
import * as compoundAnalysisActions from "../actions";
import { haveDisplayResult } from "../helpers";
import { CompoundAnalysisProps } from "../types";
import useDefaultFrameView from "./DefaultFrameView";
import AnalysisLayoutTwoCol from "./layouts/AnalysisLayoutTwoCol";
import Toolbar from "./Toolbar";

const CenterOfMassAnalysis: React.FC<CompoundAnalysisProps> = ({ compoundAnalysis, dataset }) => {
    const { shape } = dataset.params;
    const [scanHeight, scanWidth, imageHeight, imageWidth] = shape;
    const minLength = Math.min(imageWidth, imageHeight);
    const [cx, setCx] = useState(imageWidth / 2);
    const [cy, setCy] = useState(imageHeight / 2);
    const [r, setR] = useState(minLength / 4);
    const [flip_y, setFlipY] = useState(false);
    const [scan_rotation, setScanRotation] = useState("0.0");

    const dispatch = useDispatch();

    const rHandle = {
        x: cx - r,
        y: cy,
    }

    const handleCenterChange = defaultDebounce((newCx: number, newCy: number) => {
        setCx(newCx);
        setCy(newCy);
    });
    const handleRChange = defaultDebounce(setR);

    const frameViewHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={cx} y={cy}
            imageWidth={imageWidth}
            onDragMove={handleCenterChange}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={inRectConstraint(imageWidth, imageHeight)} />
        <DraggableHandle x={rHandle.x} y={rHandle.y}
            imageWidth={imageWidth}
            onDragMove={cbToRadius(cx, cy, handleRChange)}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={keepOnCY(cy)} />
    </>);

    const frameViewWidgets = (
        <Disk cx={cx} cy={cy} r={r}
            imageWidth={imageWidth} imageHeight={imageHeight} />
    )

    const {
        frameViewTitle, frameModeSelector,
        handles: resultHandles, widgets: resultWidgets
    } = useDefaultFrameView({
        scanWidth,
        scanHeight,
        compoundAnalysisId: compoundAnalysis.compoundAnalysis,
        doAutoStart: compoundAnalysis.doAutoStart,
    })

    const subtitle = <>{frameViewTitle} Disk: center=(x={cx.toFixed(2)}, y={cy.toFixed(2)}), r={r.toFixed(2)}</>;

    let parsedScanRotation: number =  parseFloat(scan_rotation);
    if (!parsedScanRotation) {
        parsedScanRotation = 0.0;
    }

    const runAnalysis = () => {
        dispatch(compoundAnalysisActions.Actions.run(compoundAnalysis.compoundAnalysis, 1, {
            analysisType: AnalysisTypes.CENTER_OF_MASS,
            parameters: {
                shape: "com",
                cx,
                cy,
                r,
                flip_y,
                scan_rotation: parsedScanRotation,
            }
        }));
    };

    const analyses = useSelector((state: RootReducer) => state.analyses)
    const jobs = useSelector((state: RootReducer) => state.jobs)

    const haveResult = haveDisplayResult(
        compoundAnalysis,
        analyses,
        jobs,
        [1],
    );

    // NOTE: haveResult is not a dependency here, as we don't want to re-run directly
    // after the results have become available.
    React.useEffect(() => {
        if (haveResult) {
            runAnalysis();
        }
    }, [flip_y, scan_rotation]);

    const toolbar = <Toolbar compoundAnalysis={compoundAnalysis} onApply={runAnalysis} busyIdxs={[1]} />

    const updateFlipY = (e: React.ChangeEvent<HTMLInputElement>, { checked }: { checked: boolean }) => {
        setFlipY(checked);
    };

    const updateScanRotation = (e: React.ChangeEvent<HTMLInputElement>, { value }: { value: string }) => {
        if (value === "-") {
            setScanRotation("-");
        }
        setScanRotation(value);
    };

    // TODO: debounce parameters
    const comParams = (
        <>
            <Header>
                <Modal trigger={
                    <Header.Content>
                        Parameters
                        {' '}
                        <Icon name="info circle" size="small" link />
                    </Header.Content>
                }>
                    <Popup.Header>CoM / first moment parameters</Popup.Header>
                    <Popup.Content>
                        <Header>Flip in y direction</Header>
                        <p>
                        Flip the Y coordinate. Some detectors, for example Quantum
                        Detectors Merlin, may have pixel (0, 0) at the lower
                        left corner. This has to be corrected to get the sign of
                        the y shift as well as curl and divergence right.
                        </p>
                        <Header>Rotation between scan and detector</Header>
                        <p>
                            The optics of an electron microscope can rotate the
                            image. Furthermore, scan generators may allow
                            scanning in arbitrary directions. This means that
                            the x and y coordinates of the detector image are
                            usually not parallel to the x and y scan
                            coordinates. For interpretation of center of mass
                            shifts, however, the shift vector in detector
                            coordinates has to be put in relation to the
                            position on the sample. This parameter can be used
                            to rotate the detector coordinates to match the scan
                            coordinate system. A positive value rotates the
                            displacement vector clock-wise. That means if the
                            detector seems rotated to the right relative to the
                            scan, this value should be negative to counteract
                            this rotation.
                        </p>
                    </Popup.Content>
                </Modal>
            </Header>
            <Form>
                <Form.Field control={Checkbox} label="Flip in y direction" checked={flip_y} onChange={updateFlipY} />
                <Form.Field type="number" control={Input} label="Rotation between scan and detector (deg)" value={scan_rotation} onChange={updateScanRotation} />
                <Form.Field type="range" min="-180" max="180" step="0.1" control={Input} value={scan_rotation} onChange={updateScanRotation} />
            </Form>
        </>
    );

    return (
        <AnalysisLayoutTwoCol
            title="CoM / first moment analysis" subtitle={subtitle}
            left={<>
                <ResultList
                    extraHandles={frameViewHandles} extraWidgets={frameViewWidgets}
                    analysisIndex={0} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                    width={imageWidth} height={imageHeight}
                    selectors={frameModeSelector}
                />
            </>}
            right={<>
                <ResultList
                    analysisIndex={1} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                    width={scanWidth} height={scanHeight}
                    extraHandles={resultHandles}
                    extraWidgets={resultWidgets}
                />
            </>}
            toolbar={toolbar}
            params={comParams}
        />
    );
}

export default CenterOfMassAnalysis;
