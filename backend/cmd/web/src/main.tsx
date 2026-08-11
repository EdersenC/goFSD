import {CssBaseline, GlobalStyles, ThemeProvider} from "@mui/material";
import {StrictMode} from "react";
import {createRoot} from "react-dom/client";
import {App} from "./App";
import {operatorTheme} from "./theme";

const rootElement = document.getElementById("root");
if (!rootElement) {
    throw new Error("Stop Sign Lab root element is missing");
}

createRoot(rootElement).render(
    <StrictMode>
        <ThemeProvider theme={operatorTheme}>
            <CssBaseline />
            <GlobalStyles styles={{
                "*": {boxSizing: "border-box"},
                "body": {margin: 0},
                "::selection": {background: "rgba(233,255,91,.28)"},
            }} />
            <App />
        </ThemeProvider>
    </StrictMode>,
);
