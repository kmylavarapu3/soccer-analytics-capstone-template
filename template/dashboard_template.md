# Analytics Dashboard Template

This folder contains a modern, interactive analytics dashboard built with **Plotly Dash** to help you get started with soccer data visualization. The dashboard features a refined dark theme, responsive design, and dynamic filtering capabilities.

## Dashboard Features

### Overview Statistics Cards
Four key metric cards that update dynamically based on filter selections:
- **Total Matches**: Count of matches matching current filters
- **Total Events**: Count of events from filtered matches
- **Unique Players**: Number of unique players in filtered matches
- **Average Goals per Match**: Mean goals across filtered matches

### Interactive Visualizations
- **Event Type Distribution**: Bar chart showing the most common event types (Pass, Shot, Carry, etc.)
- **Match Results Distribution**: Pie chart showing Home Wins vs Away Wins vs Draws
- **xG Distribution**: Histogram of expected goals (xG) values for all shots
- **Goals by Competition**: Bar chart showing average goals per match across competitions

### Advanced Filtering
- **Competition Filter**: Filter by specific competitions (e.g., La Liga, Premier League)
- **Season Filter**: Filter by season (e.g., 2023/2024)
- **Team Filter**: Filter by specific teams (home or away)
- **Searchable Dropdowns**: All filters support search functionality for quick selection
- **Dynamic Updates**: All charts and statistics update in real-time based on filter combinations

## Design & Styling

### Color Scheme
The dashboard uses a professional dark indigo color palette:
- **Background**: `#1A1B3A` (Dark Indigo) - Reduces eye strain and allows accents to pop
- **Primary Accent**: `#F49D52` (Tangerine Orange) - Used for primary actions and highlights
- **Secondary Accent**: `#759ACE` (Soft Cornflower) - Used for secondary actions
- **Primary Text**: `#FFFFFF` (Pure White) - Main text and headings
- **Secondary Text**: `#BFC0C5` (Cool Silver) - Muted text and captions

### Typography
- **Headings & Body**: DM Sans (Google Fonts) - Clean, modern sans-serif
- **Statistics Numbers**: JetBrains Mono (Google Fonts) - Monospace font for better number readability

### User Experience Features
- **Loading States**: All charts wrapped with loading spinners during data updates
- **Hover Effects**: Interactive cards with smooth hover transitions
- **Responsive Design**: Adapts to tablet and mobile screen sizes
- **Smooth Animations**: Fade-in animations for stats cards on page load
- **Accessibility**: Focus states and keyboard navigation support

## Running the Dashboard

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure data is downloaded**:
   ```bash
   python data/download_data.py
   ```

3. **Run the dashboard**:
   ```bash
   python template/dashboard_template.py
   ```

4. **Open your browser** to: `http://127.0.0.1:8050`

The dashboard will automatically load StatsBomb data from the `data/Statsbomb/` directory.

## Architecture & Code Structure

### File Organization
```
template/
├── dashboard_template.py    # Main dashboard application
├── dashboard_template.md     # This documentation file
└── assets/
    └── styles.css           # Custom CSS for styling and animations
```

### Key Components

**Theme Configuration**
- Centralized `THEME` dictionary defines colors, spacing, shadows, and transitions
- Consistent styling across all components
- Easy to customize by modifying the THEME config

**Custom Plotly Template**
- Pre-configured Plotly template matching the dashboard theme
- Consistent grid styling, colors, and fonts across all charts
- Applied automatically to all visualizations

**Reusable Style Dictionaries**
- `CARD_STYLE`: Base card styling
- `STATS_CARD_STYLE`: Statistics card styling with animations
- `HEADER_STYLE`: Consistent header typography
- `LABEL_STYLE`: Form label styling

**Callbacks**
- `update_stats_cards()`: Updates all four statistics cards based on filters
- `update_results_chart()`: Updates match results pie chart
- `update_goals_chart()`: Updates goals by competition bar chart

### Data Processing
- Uses **Polars** for high-performance data processing
- Efficient filtering and aggregation operations
- Lazy evaluation for optimal performance

## Customization Guide

### Changing Colors
Edit the `THEME` dictionary in `dashboard_template.py`:
```python
THEME = {
    "colors": {
        "background": "#1A1B3A",  # Change background color
        "accent": "#F49D52",      # Change primary accent
        # ... etc
    }
}
```

### Adding New Charts
1. Create a new card div with `CARD_STYLE`
2. Add a `dcc.Graph` component with an `id`
3. Create a callback function that filters data and returns a Plotly figure
4. Use `plotly_template` for consistent styling

### Adding New Filters
1. Add a new `dcc.Dropdown` in the filters section
2. Update callback functions to accept the new filter input
3. Add filtering logic in the callback

### Customizing CSS
Edit `assets/styles.css` to modify:
- Animations and transitions
- Responsive breakpoints
- Dropdown styling
- Card hover effects

## Key Technologies

- **Plotly Dash**: Web framework for building analytical dashboards
- **Plotly Express & Graph Objects**: Chart creation and customization
- **Polars**: High-performance data processing (consistent with EDA template)
- **StatsBomb Open Data**: Professional soccer event data
- **Google Fonts**: DM Sans and JetBrains Mono typography

## Learning Resources

- [Plotly Dash Documentation](https://dash.plotly.com/)
- [Plotly Graph Objects](https://plotly.com/python/)
- [Plotly Express](https://plotly.com/python/plotly-express/)
- [Polars Documentation](https://pola-rs.github.io/polars/)
- [StatsBomb Open Data](https://github.com/statsbomb/open-data)

## Future Enhancement Ideas

This template is designed to be extended. Consider adding:

- **Player-specific views**: Individual player performance metrics and comparisons
- **Pitch visualizations**: Shot maps, pass networks, heatmaps using Plotly
- **Polymarket integration**: Overlay betting market data with match events
- **Advanced metrics**: xThreat, PPDA, progressive passes, field tilt
- **Time-series analysis**: Performance trends over seasons
- **Team comparisons**: Head-to-head statistics and team performance matrices
- **Match detail views**: Drill-down into individual match events
- **Export functionality**: Download charts and filtered data as CSV/PNG

## Notes

- The dashboard loads all data into memory on startup for fast filtering
- For very large datasets, consider implementing pagination or lazy loading
- All filters work together - selecting multiple filters narrows the dataset
- Statistics cards update automatically when filters change
- The dashboard is fully responsive and works on mobile devices

---

**Note**: This is a production-ready template with modern styling and UX best practices. Fork it, customize it, and make it your own!
