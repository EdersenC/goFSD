import {createTheme} from "@mui/material/styles";

export const operatorTheme = createTheme({
    palette: {
        mode: "dark",
        primary: {main: "#e9ff5b", contrastText: "#11140d"},
        secondary: {main: "#57d9ff", contrastText: "#071318"},
        error: {main: "#ff5c5c"},
        success: {main: "#57db8a"},
        warning: {main: "#ffbd59"},
        background: {default: "#080b0e", paper: "#10151a"},
        divider: "rgba(213, 227, 238, .14)",
        text: {primary: "#edf4f7", secondary: "#94a5ae"},
    },
    shape: {borderRadius: 10},
    typography: {
        fontFamily: 'Inter, "IBM Plex Sans", system-ui, sans-serif',
        h1: {fontSize: "clamp(1.65rem, 3vw, 2.45rem)", fontWeight: 760, letterSpacing: "-.035em"},
        h2: {fontSize: "1.05rem", fontWeight: 760, letterSpacing: "-.01em"},
        overline: {fontSize: ".68rem", fontWeight: 800, letterSpacing: ".13em"},
        button: {fontWeight: 780, textTransform: "none"},
    },
    components: {
        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundImage: "none",
                    border: "1px solid rgba(213, 227, 238, .12)",
                    boxShadow: "0 18px 45px rgba(0,0,0,.18)",
                },
            },
        },
        MuiButton: {defaultProps: {disableElevation: true}},
        MuiTextField: {defaultProps: {size: "small"}},
    },
});
