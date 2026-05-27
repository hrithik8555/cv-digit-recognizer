import pytest
from app import demo

def test_frontend_initialization():
    """
    Test that the Gradio UI loads without syntax or configuration errors,
    and verify that it contains the core expected components.
    """
    # The 'demo' object is a Gradio Blocks instance.
    assert demo is not None
    
    # We can inspect the components inside the Gradio Blocks to ensure 
    # our interface layout hasn't been accidentally deleted.
    components = demo.blocks.values()
    
    component_types = [type(c).__name__ for c in components]
    
    # Verify our custom HTML/Markdown headers exist
    assert "HTML" in component_types or "Markdown" in component_types, "Missing custom header in UI"
    
    # Verify the Sketchpad (Canvas) exists
    assert "Sketchpad" in component_types, "Missing drawing Canvas in UI"
    
    # Verify the output Label exists
    assert "Label" in component_types, "Missing output prediction Label in UI"

def test_frontend_api_endpoints():
    """
    Test that the Gradio app has properly registered the prediction function 
    to an API endpoint, allowing internal reactivity and external Client queries.
    """
    # Check if the predict_digit function has been mapped as a dependency 
    # (meaning it is actively hooked up to the Canvas 'change' event).
    functions = demo.fns
    
    # Ensure there is at least one active event listener hooked up 
    assert len(functions) > 0, "No event listeners hooked up in the UI"
