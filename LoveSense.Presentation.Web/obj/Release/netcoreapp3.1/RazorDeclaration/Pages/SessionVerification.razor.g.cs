#pragma checksum "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\SessionVerification.razor" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "fea5e5529e2c4f4772cf8dffa0884d6629e767e5"
// <auto-generated/>
#pragma warning disable 1591
#pragma warning disable 0414
#pragma warning disable 0649
#pragma warning disable 0169

namespace LoveSense.Presentation.Web.Pages
{
    #line hidden
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.AspNetCore.Components;
#nullable restore
#line 1 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using System.Net.Http;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Components.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 4 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Components.Forms;

#line default
#line hidden
#nullable disable
#nullable restore
#line 5 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Components.Routing;

#line default
#line hidden
#nullable disable
#nullable restore
#line 6 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Components.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 7 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.JSInterop;

#line default
#line hidden
#nullable disable
#nullable restore
#line 8 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using LoveSense.Presentation.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 9 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using LoveSense.Presentation.Web.Shared;

#line default
#line hidden
#nullable disable
    [Microsoft.AspNetCore.Components.RouteAttribute("/sessions")]
    public partial class SessionVerification : Microsoft.AspNetCore.Components.ComponentBase
    {
        #pragma warning disable 1998
        protected override void BuildRenderTree(Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder)
        {
        }
        #pragma warning restore 1998
#nullable restore
#line 35 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\SessionVerification.razor"
       
    private IEnumerable<SessionVerification> sessions = Array.Empty<SessionVerification>();

    protected override async Task OnInitializedAsync()
    {
        sessions = await SessionVerificator.GetSessionsAsync();
    }

#line default
#line hidden
#nullable disable
        [global::Microsoft.AspNetCore.Components.InjectAttribute] private ISessionVerificator SessionVerificator { get; set; }
    }
}
#pragma warning restore 1591
