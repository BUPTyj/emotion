$( window ).on( "load", function() {
    // init Isotope bleuket load more
    var $container = $('.isotope-items-wrap').isotope({
      itemSelector: '.isotope-item',
         transitionDuration: '0.5s',
         masonry: {
          columnWidth: '.grid-sizer'
         }
    });
    //****************************
    // Isotope Load more button
    //****************************
    var initShow = 6; //number of images loaded on init & onclick load more button
    var counter = initShow; //counter for load more button
    var iso = $container.data('isotope'); // get Isotope instance
    loadMore(initShow); //execute function onload
    function loadMore(toShow) {
      $container.find(".hidden").removeClass("hidden");
      var hiddenElems = iso.filteredItems.slice(toShow, iso.filteredItems.length).map(function(item) {
      return item.element;
      });
      $(hiddenElems).addClass('hidden');
      $container.isotope('layout');
    
      //when no more to load, hide show more button
      if (hiddenElems.length == 0) {
      $("#load-more").hide();
      }
      else {
      $("#load-more").show();
      };
    }
    //append load more button
    $container.after('<div class="margin-top-30 text-center"><button id="load-more" class="mt30  sw-btn sw-blue-btn"> Load More</button></div>');
    //when load more button clicked
    $("#load-more").click(function() {
      if ($('#filters').data('clicked')) {
      //when filter button clicked, set initial value for counter
      counter = initShow;
      j$('#filters').data('clicked', false);
      } else {
      counter = counter;
      };
      counter = counter + initShow;
      loadMore(counter);
    });
});